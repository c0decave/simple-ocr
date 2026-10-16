"""Model benchmark: run one image through several models, each one alone on a clean model server.

Before every model is loaded, everything that is loaded gets unloaded, so load times and memory use
are comparable (LM Studio, Ollama). vLLM serves fixed models, and with model loading disabled
(backends.MODEL_LOADING_ALLOWED) only loaded models are used: then the models only run, one after
the other, and no load time is measured. Results are compared with app.compare.
"""
import json
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path

from app import backends, compare
from app import server as host  # circular import: only used at call time

UNLOAD_VERIFY_ATTEMPTS = 20   # how often to look whether the unload really took effect
UNLOAD_VERIFY_DELAY = 0.5     # seconds between those looks
MAX_LOG_LINES = 300


def unload_all(config):
    """Unload everything that is loaded and confirm that nothing is left. Returns the unloaded ids."""
    ids = backends.loaded_ids(config)
    for ident in ids:
        try:
            backends.unload(config, ident)
        except host.ApiError as error:
            raise host.ApiError(502, "Could not unload %s: %s" % (ident, error.message))
    for attempt in range(UNLOAD_VERIFY_ATTEMPTS):
        remaining = backends.loaded_ids(config)
        if not remaining:
            return ids
        time.sleep(UNLOAD_VERIFY_DELAY)
    raise host.ApiError(502, "Still loaded after unload: %s" % ", ".join(remaining))


RUN_ID = re.compile(r"^[0-9a-f]{12}$")


class SavedRun:
    """A finished run read back from disk: same read interface as Run, nothing to cancel."""

    ANALYSIS_KEYS = ("similar_threshold", "matrix", "tree", "agreement", "pairs", "groups")

    def __init__(self, data):
        data = dict(data)
        analysis = data.get("analysis")
        if not isinstance(analysis, dict) or any(key not in analysis for key in self.ANALYSIS_KEYS):
            # files written before the tree and agreement existed (or hand-made exports): rebuild
            texts = {r["model"]: r.get("text") or "" for r in data["results"] if r.get("status") == "ok"}
            data["analysis"] = compare.analyze(texts)
        # every field the page reads, also for files that lack them
        results = data["results"]
        data.setdefault("mode", "ocr")
        data.setdefault("step", "finished")
        data.setdefault("total", len(results))
        data.setdefault("done", len(results))
        data.setdefault("elapsed_seconds", None)
        if not isinstance(data.get("log"), list):
            data["log"] = []
        if not isinstance(data.get("restore"), dict):
            data["restore"] = {"requested": False, "models": [], "loaded": [], "failed": []}
        self.data = data
        self.id = data["id"]
        self.status = data.get("status")

    def snapshot(self):
        return json.loads(json.dumps(self.data))

    def result_text(self, model):
        for result in self.data.get("results", []):
            if result.get("model") == model and result.get("status") == "ok":
                return result.get("text")
        return None

    def cancel(self):
        pass


class RunStore:
    """Finished runs as <data_dir>/runs/<id>.json. An empty data_dir switches persistence off."""

    def __init__(self, data_dir):
        self.dir = Path(data_dir) / "runs" if data_dir else None

    @property
    def enabled(self):
        return self.dir is not None

    def save(self, snapshot):
        if not self.enabled:
            return
        temp = None
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            target = self.dir / (snapshot["id"] + ".json")
            temp = target.with_suffix(".tmp")
            temp.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
            os.replace(temp, target)
        except (OSError, ValueError, TypeError) as error:   # a full or read-only disk must not turn a finished run into an error
            print("simple-ocr: could not save run %s: %s" % (snapshot.get("id"), error), file=sys.stderr, flush=True)
            if temp is not None:
                temp.unlink(missing_ok=True)

    def _read(self, path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("id") != path.stem or not isinstance(data.get("results"), list):
            return None
        if not all(isinstance(r, dict) and isinstance(r.get("model"), str)
                   and isinstance(r.get("text"), (str, type(None))) for r in data["results"]):
            return None
        if not isinstance(data.get("started_at", 0), (int, float)) or isinstance(data.get("started_at"), bool):
            return None
        return data

    def load(self, run_id):
        if not self.enabled or not RUN_ID.match(run_id):
            return None
        data = self._read(self.dir / (run_id + ".json"))
        return SavedRun(data) if data else None

    def list(self):
        """Summaries, newest first. Files that are not valid runs are skipped."""
        if not self.enabled or not self.dir.is_dir():
            return []
        found = []
        for path in self.dir.glob("*.json"):
            if not RUN_ID.match(path.stem):
                continue
            data = self._read(path)
            if data:
                results = data["results"]
                found.append({
                    "id": data["id"], "started_at": data.get("started_at", 0), "status": data.get("status"),
                    "mode": data.get("mode"), "total": data.get("total", len(results)),
                    "ok": sum(1 for r in results if r.get("status") == "ok"),
                    "elapsed_seconds": data.get("elapsed_seconds"),
                })
        found.sort(key=lambda entry: entry["started_at"], reverse=True)
        return found


class Run:
    def __init__(self, config, mode, image, models, restore=False, previous=(), on_finish=None):
        self.id = uuid.uuid4().hex[:12]
        self.on_finish = on_finish
        self.config, self.mode, self.image, self.models = config, mode, image, models
        self.restore, self.previous = restore, list(previous)
        self.restored, self.restore_failed = [], []
        self.lock = threading.RLock()   # re-entered by snapshot() while the final state is written
        self.cancel_requested = threading.Event()
        self.status, self.step, self.current = "running", "starting", None
        self._outcome = "done"   # becomes the status only after cleanup and restore are finished
        self.error, self.analysis = None, None
        self.results, self.log = [], []
        self.started_at = time.time()
        self._started = time.monotonic()
        self._step_started = time.monotonic()
        self._finished = None

    # ---- state shared with the HTTP threads
    def _set(self, **fields):
        with self.lock:
            for key, value in fields.items():
                setattr(self, key, value)
            if "step" in fields:
                self._step_started = time.monotonic()

    def _note(self, message):
        with self.lock:
            self.log.append({"t": round(time.monotonic() - self._started, 1), "message": message})
            del self.log[:-MAX_LOG_LINES]

    def snapshot(self):
        with self.lock:
            now = self._finished or time.monotonic()
            return {
                "id": self.id, "status": self.status, "mode": self.mode, "step": self.step,
                "current_model": self.current, "total": len(self.models), "done": len(self.results),
                "error": self.error, "analysis": self.analysis, "started_at": self.started_at,
                "restore": {"requested": self.restore, "models": list(self.previous),
                            "loaded": list(self.restored), "failed": list(self.restore_failed)},
                "elapsed_seconds": round(now - self._started, 1),
                "step_elapsed_seconds": round(time.monotonic() - self._step_started, 1) if self._finished is None else 0,
                "results": [dict(r) for r in self.results], "log": list(self.log),
            }

    def result_text(self, model):
        with self.lock:
            for result in self.results:
                if result["model"] == model and result["status"] == "ok":
                    return result["text"]
        return None

    def cancel(self):
        self.cancel_requested.set()

    # ---- the worker
    def execute(self):
        try:
            for meta in self.models:
                if self.cancel_requested.is_set():
                    break
                self._run_model(meta)
            if self.cancel_requested.is_set():   # also when the cancel came during the last model
                self._outcome = "cancelled"
        except host.ApiError as error:   # unload problems: the host state is unknown, stop here
            self._outcome = "error"
            self._set(error=error.message)
            self._note("Stopped: " + error.message)
        except Exception as error:  # noqa: BLE001 - the worker must always end in a final state
            self._outcome = "error"
            self._set(error="Internal error: %s: %s" % (type(error).__name__, error))
        finally:
            self._finish()

    def _run_model(self, meta):
        model = meta["id"]
        managed = backends.manages_models(self.config)
        self._set(current=model, step="unloading" if managed else "inferring")
        if managed:
            started = time.monotonic()
            unloaded = unload_all(self.config)
            self._note("%s: unloaded %s in %.1f s" % (model, ", ".join(unloaded) or "nothing", time.monotonic() - started))
        result = {
            "model": model, "size_bytes": meta.get("size_bytes"), "params": meta.get("params"),
            "quantization": meta.get("quantization"), "status": "error", "error": None, "text": "", "chars": 0,
            "load_seconds": None, "reported_load_seconds": None, "infer_seconds": None, "total_seconds": None,
        }
        try:
            if managed:
                self._set(step="loading")
                started = time.monotonic()
                result["reported_load_seconds"] = backends.load(self.config, model)
                result["load_seconds"] = round(time.monotonic() - started, 2)
                self._note("%s: loaded in %.2f s" % (model, result["load_seconds"]))
            if self.cancel_requested.is_set():
                result["error"] = "cancelled before inference"
            else:
                self._set(step="inferring")
                started = time.monotonic()
                answer = host.analyze(self.config, self.image, self.mode, model)
                result["infer_seconds"] = round(time.monotonic() - started, 2)
                result["total_seconds"] = round((result["load_seconds"] or 0) + result["infer_seconds"], 2)
                result.update(status="ok", text=answer["text"], chars=len(answer["text"]))
                self._note("%s: answered in %.2f s, %d characters" % (model, result["infer_seconds"], result["chars"]))
        except host.ApiError as error:
            result["error"] = error.message
            self._note("%s: failed: %s" % (model, error.message))
        with self.lock:
            self.results.append(result)

    def _restore_previous(self):
        """Load the models that were loaded before the run, one after the other."""
        self._set(step="restoring")
        for model in self.previous:
            self._set(current=model)
            started = time.monotonic()
            try:
                backends.load(self.config, model)
                with self.lock:
                    self.restored.append(model)
                self._note("Restored %s in %.1f s" % (model, time.monotonic() - started))
            except host.ApiError as error:
                with self.lock:
                    self.restore_failed.append({"model": model, "error": error.message})
                self._note("Could not restore %s: %s" % (model, error.message))

    def _finish(self):
        """Clean up, then always reach the final state: a run left "running" would block the app until a restart."""
        try:
            self._clean_up()
        except Exception as error:  # noqa: BLE001 - e.g. llama.cpp's can_load() asks a server that just failed
            self._note("Cleanup stopped: %s" % getattr(error, "message", error))
        texts = {r["model"]: r["text"] for r in self.results if r["status"] == "ok"}
        try:
            analysis = compare.analyze(texts)
        except Exception as error:  # noqa: BLE001 - same reason
            analysis = None
            self._note("Comparison failed: %s" % error)
        with self.lock:
            self.analysis = analysis
            self.status = self._outcome
            self.step, self.image = "finished", None
            self._finished = time.monotonic()
            if self.on_finish:   # written before the final status becomes visible to pollers
                self.on_finish(self.snapshot())

    def _clean_up(self):
        clean = False
        if not self.config.allow_loading:
            self._note("Model loading is disabled: nothing was unloaded or loaded")
        elif not backends.can_load(self.config):
            self._note("%s serves fixed models: nothing to unload" % self.config.backend_name)
        else:
            try:
                self._set(step="cleaning up", current=None)
                unload_all(self.config)       # leave the server with nothing loaded
                clean = True
                self._note("Unloaded the last model, nothing is loaded now")
            except host.ApiError as error:
                self._note("Cleanup failed: " + error.message)
        if self.restore and clean:        # never load on top of models that could not be unloaded
            try:
                self._restore_previous()
            except Exception as error:  # noqa: BLE001 - the final state must always be reached
                self._note("Restore stopped: %s" % error)


class Runs:
    """All runs of this server. Only one may be active, because a run changes what is loaded."""

    def __init__(self, data_dir=""):
        self.lock = threading.Lock()
        self.by_id, self.latest = {}, None
        self.analyzing = 0   # /api/analyze requests in flight: a run would unload their model
        self.store = RunStore(data_dir)

    def active(self):
        with self.lock:
            return self.latest is not None and self.latest.status == "running"

    def begin_analysis(self):
        """Claim a slot for one /api/analyze request; refused while a run is active (it changes what is loaded)."""
        with self.lock:
            if self.latest is not None and self.latest.status == "running":
                raise host.ApiError(409, "A benchmark run is in progress; try again when it has finished")
            self.analyzing += 1

    def end_analysis(self):
        with self.lock:
            self.analyzing -= 1

    def start(self, config, mode, image, models, restore=False, previous=()):
        with self.lock:
            if self.latest is not None and self.latest.status == "running":
                raise host.ApiError(409, "A benchmark run is already in progress")
            if self.analyzing:
                raise host.ApiError(409, "An image is being analyzed; start the run when it has finished")
            run = Run(config, mode, image, models, restore, previous, on_finish=self.store.save)
            self.by_id[run.id] = run
            self.latest = run
        threading.Thread(target=run.execute, daemon=True).start()
        return run

    def get(self, run_id):
        with self.lock:
            run = self.latest if run_id == "latest" else self.by_id.get(run_id)
        if run is not None:
            return run
        if run_id == "latest":
            saved = self.store.list()
            run_id = saved[0]["id"] if saved else ""
        return self.store.load(run_id)

    def saved(self):
        return {"enabled": self.store.enabled, "runs": self.store.list()}
