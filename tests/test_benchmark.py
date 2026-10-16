"""Tests for the model benchmark run against a stateful fake LM Studio."""
import base64
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from app import backends, benchmark, server

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
INVOICE = "Rechnung Nr. 2026-0417\nMüller & Söhne GmbH\nSumme brutto: 68,85 €"
GB = 1000 ** 3


class FakeLmStudio:
    """Keeps the set of loaded models and records every load and unload in order."""

    def __init__(self, models, loaded=()):
        self.models = models          # key -> dict(vision, size, params, quant, text, load, chat, delay)
        self.loaded = list(loaded)
        self.events = []              # ("unload", id) / ("load", key, loaded_before) / ("chat", key)
        self.unload_fails = set()     # ids whose unload answers HTTP 500
        self.unload_ignored = False   # unload answers 200 but nothing is unloaded
        self.auth = []
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                outer.auth.append(self.headers.get("Authorization"))
                if self.path == "/api/v1/models":
                    with outer.lock:
                        listing = [{
                            "key": key, "type": "llm", "size_bytes": spec.get("size", GB),
                            "params_string": spec.get("params", "2B"),
                            "quantization": {"name": spec.get("quant", "Q4_K_M")},
                            "capabilities": {"vision": spec.get("vision", True)},
                            "loaded_instances": [{"id": key}] if key in outer.loaded else [],
                        } for key, spec in outer.models.items()]
                    return self._reply(200, {"models": listing})
                self._reply(404, {"error": "no route"})

            def do_POST(self):
                outer.auth.append(self.headers.get("Authorization"))
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if self.path == "/api/v1/models/load":
                    time.sleep(outer.models.get(body.get("model"), {}).get("load_delay", 0))
                with outer.lock:
                    if self.path == "/api/v1/models/unload":
                        ident = body.get("instance_id")
                        if ident not in outer.loaded:
                            return self._reply(404, {"error": {"type": "model_not_found", "message": "not loaded"}})
                        if ident in outer.unload_fails:
                            return self._reply(500, {"error": {"type": "unload_failed", "message": "busy"}})
                        outer.events.append(("unload", ident))
                        if not outer.unload_ignored:
                            outer.loaded.remove(ident)
                        return self._reply(200, {"instance_id": ident})
                    if self.path == "/api/v1/models/load":
                        key = body.get("model")
                        spec = outer.models.get(key)
                        outer.events.append(("load", key, tuple(outer.loaded)))
                        if spec is None:
                            return self._reply(404, {"error": {"type": "model_not_found", "message": "unknown"}})
                        if spec.get("load") == "http_error":
                            return self._reply(500, {"error": {"type": "model_load_failed", "message": "out of memory"}})
                        if spec.get("load") == "soft_error":
                            return self._reply(200, {"error": "Unexpected endpoint or method"})
                        outer.loaded.append(key)
                        return self._reply(200, {"type": "llm", "instance_id": key,
                                                 "load_time_seconds": 0.25, "status": "loaded"})
                    if self.path == "/v1/chat/completions":
                        key = body.get("model")
                        spec = outer.models.get(key, {})
                        if key not in outer.loaded:
                            return self._reply(400, {"error": "model not loaded"})
                        outer.events.append(("chat", key))
                delay = spec.get("delay", 0)
                if delay:
                    time.sleep(delay)
                if spec.get("chat") == "http_error":
                    return self._reply(500, {"error": "generation failed"})
                self._reply(200, {"choices": [{"message": {"content": spec.get("text", INVOICE)}}]})

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def loads(self):
        return [e for e in self.events if e[0] == "load"]


class BenchmarkTestCase(unittest.TestCase):
    MODELS = {
        "m1": {"text": INVOICE, "size": 2 * GB},
        "m2": {"text": INVOICE, "size": 5 * GB},                                   # identical to m1
        "m3": {"text": INVOICE.replace("Müller", "Miller"), "size": 7 * GB},        # similar
        "m4": {"text": "A cat sits on a red sofa by the window.", "size": 3 * GB},  # different
        "text-llm": {"vision": False, "size": 4 * GB},
    }

    def setUp(self):
        self._saved = (benchmark.UNLOAD_VERIFY_ATTEMPTS, benchmark.UNLOAD_VERIFY_DELAY)
        benchmark.UNLOAD_VERIFY_ATTEMPTS, benchmark.UNLOAD_VERIFY_DELAY = 3, 0.01
        self.lm = FakeLmStudio({k: dict(v) for k, v in self.MODELS.items()}, loaded=["text-llm", "m2"])
        self.apps = []

    def tearDown(self):
        benchmark.UNLOAD_VERIFY_ATTEMPTS, benchmark.UNLOAD_VERIFY_DELAY = self._saved
        self.lm.close()
        for app in self.apps:
            app.shutdown()
            app.server_close()

    def start_app(self, **config):
        # these tests cover the loading mode; the hard-coded default (off) is tested in test_backends
        settings = dict(backend_url=self.lm.url, timeout=5, allow_loading=True)
        settings.update(config)
        app = server.make_server(server.Config(**settings), "127.0.0.1", 0)
        threading.Thread(target=app.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        self.apps.append(app)
        return "http://127.0.0.1:%d" % app.server_address[1]

    def request(self, base, path, body=None, method=None):
        req = urllib.request.Request(base + path, data=body, method=method or ("POST" if body is not None else "GET"))
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def start_run(self, base, models=None, mode="ocr", image=PNG, all_models=False, restore=False):
        query = [("mode", mode)] + [("model", m) for m in (models or [])]
        if all_models:
            query.append(("all", "1"))
        if restore:
            query.append(("restore", "1"))
        return self.request(base, "/api/debug/runs?" + urllib.parse.urlencode(query), image)

    def wait_done(self, base, run_id, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status, state = self.request(base, "/api/debug/runs/" + run_id)
            if state["status"] != "running":
                return state
            time.sleep(0.02)
        self.fail("run did not finish: %r" % state)


class OrderAndStatistics(BenchmarkTestCase):
    def run_models(self, models, **config):
        base = self.start_app(**config)
        status, started = self.start_run(base, models)
        self.assertEqual(status, 200, started)
        return base, self.wait_done(base, started["run_id"])

    def test_nothing_is_loaded_at_the_moment_any_model_is_loaded(self):
        _, state = self.run_models(["m1", "m2", "m3"])
        self.assertEqual(state["status"], "done")
        loads = self.lm.loads()
        self.assertEqual([e[1] for e in loads], ["m1", "m2", "m3"])
        self.assertTrue(all(e[2] == () for e in loads), "something was still loaded: %r" % (loads,))

    def test_initially_loaded_models_are_unloaded_first_and_nothing_is_left_at_the_end(self):
        self.run_models(["m1"])
        first_load = next(i for i, e in enumerate(self.lm.events) if e[0] == "load")
        unloaded_before = {e[1] for e in self.lm.events[:first_load] if e[0] == "unload"}
        self.assertEqual(unloaded_before, {"text-llm", "m2"})
        self.assertEqual(self.lm.loaded, [])
        self.assertEqual(self.lm.events[-1], ("unload", "m1"))

    def test_each_model_is_unloaded_before_the_next_is_loaded(self):
        self.run_models(["m1", "m3"])
        events = [e[:2] for e in self.lm.events]
        self.assertEqual(set(events[:2]), {("unload", "text-llm"), ("unload", "m2")})   # order within a batch is free
        self.assertEqual(events[2:], [("load", "m1"), ("chat", "m1"), ("unload", "m1"),
                                      ("load", "m3"), ("chat", "m3"), ("unload", "m3")])

    def test_results_carry_timings_size_and_text(self):
        _, state = self.run_models(["m1", "m4"])
        by_model = {r["model"]: r for r in state["results"]}
        r = by_model["m1"]
        self.assertEqual((r["status"], r["size_bytes"], r["params"], r["quantization"]), ("ok", 2 * GB, "2B", "Q4_K_M"))
        self.assertEqual(r["text"], INVOICE)
        self.assertEqual(r["chars"], len(INVOICE))
        self.assertGreaterEqual(r["load_seconds"], 0)
        self.assertEqual(r["reported_load_seconds"], 0.25)
        self.assertGreaterEqual(r["infer_seconds"], 0)
        self.assertAlmostEqual(r["total_seconds"], r["load_seconds"] + r["infer_seconds"], places=2)
        self.assertEqual([x["model"] for x in state["results"]], ["m1", "m4"])

    def test_analysis_groups_identical_similar_and_different_results(self):
        _, state = self.run_models(["m1", "m2", "m3", "m4"])
        groups = [g["models"] for g in state["analysis"]["groups"]]
        self.assertEqual(groups, [["m1", "m2", "m3"], ["m4"]])
        classes = {(p["a"], p["b"]): p["class"] for p in state["analysis"]["pairs"]}
        self.assertEqual(classes[("m1", "m2")], "identical")
        self.assertEqual(classes[("m1", "m3")], "similar")
        self.assertEqual(classes[("m1", "m4")], "different")

    def test_describe_mode_is_forwarded(self):
        base = self.start_app()
        _, started = self.start_run(base, ["m1"], mode="describe")
        state = self.wait_done(base, started["run_id"])
        self.assertEqual(state["mode"], "describe")

    def test_all_selects_every_vision_model_but_no_text_model(self):
        base = self.start_app()
        _, started = self.start_run(base, all_models=True)
        state = self.wait_done(base, started["run_id"])
        self.assertEqual([r["model"] for r in state["results"]], ["m2", "m1", "m3", "m4"])  # loaded first, then listing order
        self.assertNotIn("text-llm", [e[1] for e in self.lm.loads()])

    def test_api_key_is_used_for_load_and_unload(self):
        self.run_models(["m1"], api_key="tok")
        self.assertTrue(self.lm.auth and all(a == "Bearer tok" for a in self.lm.auth))


class Failures(BenchmarkTestCase):
    def finish(self, models, **config):
        base = self.start_app(**config)
        _, started = self.start_run(base, models)
        return base, self.wait_done(base, started["run_id"])

    def test_load_failure_is_recorded_and_the_run_continues(self):
        self.lm.models["m2"]["load"] = "http_error"
        _, state = self.finish(["m1", "m2", "m3"])
        self.assertEqual(state["status"], "done")
        by_model = {r["model"]: r for r in state["results"]}
        self.assertEqual(by_model["m2"]["status"], "error")
        self.assertIn("out of memory", by_model["m2"]["error"])
        self.assertEqual((by_model["m1"]["status"], by_model["m3"]["status"]), ("ok", "ok"))
        self.assertNotIn("m2", state["analysis"]["matrix"]["models"])

    def test_http_200_with_error_body_counts_as_failure(self):
        self.lm.models["m1"]["load"] = "soft_error"
        _, state = self.finish(["m1", "m3"])
        by_model = {r["model"]: r for r in state["results"]}
        self.assertEqual(by_model["m1"]["status"], "error")
        self.assertEqual(by_model["m3"]["status"], "ok")

    def test_chat_failure_is_recorded_and_the_model_is_still_unloaded(self):
        self.lm.models["m1"]["chat"] = "http_error"
        _, state = self.finish(["m1", "m3"])
        by_model = {r["model"]: r for r in state["results"]}
        self.assertEqual(by_model["m1"]["status"], "error")
        self.assertIn("generation failed", by_model["m1"]["error"])
        self.assertGreaterEqual(by_model["m1"]["load_seconds"], 0)
        self.assertTrue(all(e[2] == () for e in self.lm.loads()))

    def test_unload_failure_stops_the_run_without_loading_anything_more(self):
        self.lm.unload_fails.add("m2")
        _, state = self.finish(["m1", "m3"])
        self.assertEqual(state["status"], "error")
        self.assertIn("m2", state["error"])
        self.assertEqual(self.lm.loads(), [])

    def test_unload_that_does_not_take_effect_stops_the_run(self):
        self.lm.unload_ignored = True
        _, state = self.finish(["m1"])
        self.assertEqual(state["status"], "error")
        self.assertEqual(self.lm.loads(), [])

    def test_unreachable_lm_studio_is_a_run_error(self):
        base = self.start_app(backend_url="http://127.0.0.1:9")
        status, body = self.start_run(base, ["m1"])
        self.assertEqual(status, 502)
        self.assertIn("LM Studio", body["error"])



class RestoreAfterRun(BenchmarkTestCase):
    def finish(self, models, base=None, **kwargs):
        base = base or self.start_app()
        status, started = self.start_run(base, models, restore=True, **kwargs)
        self.assertEqual(status, 200, started)
        return base, started, self.wait_done(base, started["run_id"])

    def test_previously_loaded_models_come_back_when_asked(self):
        _, _, state = self.finish(["m1", "m3"])
        self.assertEqual(state["status"], "done")
        self.assertEqual(sorted(self.lm.loaded), ["m2", "text-llm"])
        self.assertEqual(state["restore"]["requested"], True)
        self.assertEqual(sorted(state["restore"]["loaded"]), ["m2", "text-llm"])
        self.assertEqual(state["restore"]["failed"], [])

    def test_benchmark_models_are_still_loaded_alone(self):
        self.finish(["m1", "m3"])
        benchmark_loads = [e for e in self.lm.loads() if e[1] in ("m1", "m3")]
        self.assertEqual(len(benchmark_loads), 2)
        self.assertTrue(all(e[2] == () for e in benchmark_loads))

    def test_without_the_flag_nothing_is_restored(self):
        base = self.start_app()
        _, started = self.start_run(base, ["m1"])
        state = self.wait_done(base, started["run_id"])
        self.assertEqual(self.lm.loaded, [])
        self.assertFalse(state["restore"]["requested"])

    def test_restore_failure_is_reported_and_does_not_fail_the_run(self):
        self.lm.models["text-llm"]["load"] = "http_error"
        _, _, state = self.finish(["m1"])
        self.assertEqual(state["status"], "done")
        self.assertEqual(state["restore"]["loaded"], ["m2"])
        self.assertEqual([f["model"] for f in state["restore"]["failed"]], ["text-llm"])
        self.assertIn("out of memory", state["restore"]["failed"][0]["error"])

    def test_nothing_is_restored_when_the_host_could_not_be_cleaned(self):
        self.lm.unload_fails.add("m2")
        _, _, state = self.finish(["m1"])
        self.assertEqual(state["status"], "error")
        self.assertEqual(self.lm.loads(), [])

    def test_run_stays_running_until_the_host_is_cleaned_and_restored(self):
        """Status "cancelled" or "error" must only appear after cleanup, otherwise the UI and the
        main page treat the host as free while models are still being unloaded and loaded."""
        self.lm.models["m1"]["delay"] = 0.3
        self.lm.models["m2"]["load_delay"] = 0.6          # slow restore
        base = self.start_app()
        _, started = self.start_run(base, ["m1", "m3"], restore=True)
        time.sleep(0.15)
        self.request(base, "/api/debug/runs/%s/cancel" % started["run_id"], b"")
        seen = set()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            _, state = self.request(base, "/api/debug/runs/" + started["run_id"])
            seen.add((state["status"], state["step"]))
            if state["status"] != "running":
                break
            time.sleep(0.02)
        self.assertEqual(state["status"], "cancelled")
        self.assertEqual(sorted(self.lm.loaded), ["m2", "text-llm"], "restore was not finished when the run ended")
        self.assertIsNotNone(state["analysis"])
        self.assertIn(("running", "restoring"), seen)
        self.assertNotIn(("cancelled", "restoring"), seen)

    def test_error_status_also_waits_for_cleanup(self):
        self.lm.models["m1"]["load"] = None
        self.lm.unload_fails.add("m2")
        base = self.start_app()
        _, started = self.start_run(base, ["m1"], restore=True)
        state = self.wait_done(base, started["run_id"])
        self.assertEqual(state["status"], "error")
        self.assertEqual(state["step"], "finished")
        self.assertIsNotNone(state["analysis"])

    def test_cancelled_run_still_restores(self):
        self.lm.models["m1"]["delay"] = 0.4
        base = self.start_app()
        _, started = self.start_run(base, ["m1", "m3"], restore=True)
        time.sleep(0.15)
        self.request(base, "/api/debug/runs/%s/cancel" % started["run_id"], b"")
        state = self.wait_done(base, started["run_id"])
        self.assertEqual(state["status"], "cancelled")
        self.assertEqual(sorted(self.lm.loaded), ["m2", "text-llm"])

    def test_nothing_loaded_before_means_nothing_to_restore(self):
        self.lm.loaded.clear()
        _, _, state = self.finish(["m1"])
        self.assertEqual(state["restore"]["models"], [])
        self.assertEqual(self.lm.loaded, [])


class ControlAndValidation(BenchmarkTestCase):
    def test_cancel_stops_before_the_next_model(self):
        self.lm.models["m1"]["delay"] = 0.4
        base = self.start_app()
        _, started = self.start_run(base, ["m1", "m3", "m4"])
        time.sleep(0.15)
        status, _ = self.request(base, "/api/debug/runs/%s/cancel" % started["run_id"], b"")
        self.assertEqual(status, 200)
        state = self.wait_done(base, started["run_id"])
        self.assertEqual(state["status"], "cancelled")
        self.assertEqual([e[1] for e in self.lm.loads()], ["m1"])
        self.assertEqual(self.lm.loaded, [])      # still cleaned up

    def test_cancel_during_the_last_model_ends_as_cancelled(self):
        self.lm.models["m1"]["load_delay"] = 0.4
        base = self.start_app()
        _, started = self.start_run(base, ["m1"])
        time.sleep(0.15)
        self.request(base, "/api/debug/runs/%s/cancel" % started["run_id"], b"", method="POST")
        state = self.wait_done(base, started["run_id"])
        self.assertEqual(state["status"], "cancelled")

    def test_cancel_during_the_last_inference_ends_as_cancelled(self):
        self.lm.models["m1"]["delay"] = 0.4
        base = self.start_app()
        _, started = self.start_run(base, ["m1"])
        deadline = time.monotonic() + 2
        while ("chat", "m1") not in self.lm.events and time.monotonic() < deadline:
            time.sleep(0.01)
        self.request(base, "/api/debug/runs/%s/cancel" % started["run_id"], b"", method="POST")
        state = self.wait_done(base, started["run_id"])
        self.assertEqual(state["status"], "cancelled")
        self.assertEqual(state["results"][0]["status"], "ok", "the answer that was already running is kept")

    def test_only_one_run_at_a_time_and_analyze_is_blocked_meanwhile(self):
        self.lm.models["m1"]["delay"] = 0.4
        base = self.start_app()
        _, started = self.start_run(base, ["m1"])
        status, body = self.start_run(base, ["m3"])
        self.assertEqual(status, 409)
        status, body = self.request(base, "/api/analyze", PNG)
        self.assertEqual(status, 409)
        self.assertIn("benchmark", body["error"])
        self.wait_done(base, started["run_id"])
        self.lm.loaded.append("m1")                                # the fake has no just-in-time loading
        status, _ = self.request(base, "/api/analyze", PNG)       # allowed again
        self.assertEqual(status, 200)

    def test_a_run_reaches_a_final_state_when_the_server_breaks_during_cleanup(self):
        # llama.cpp asks the server whether it can load; that call may fail in _finish as well
        base = self.start_app()
        with mock.patch.object(backends, "can_load", side_effect=backends.ApiError(502, "llama.cpp answered HTTP 500")):
            status, started = self.start_run(base, ["m1"])
            self.assertEqual(status, 200, started)
            state = self.wait_done(base, started["run_id"], timeout=5)
        self.assertEqual(state["status"], "error")
        self.assertIn("HTTP 500", state["error"])
        self.assertEqual(state["step"], "finished")
        self.assertEqual(self.start_run(base, ["m1"])[0], 200, "the next run must be possible")

    def test_an_unexpected_error_in_cleanup_still_ends_the_run(self):
        base = self.start_app()
        with mock.patch.object(benchmark, "unload_all", side_effect=[["m2", "text-llm"], RuntimeError("boom")]):
            _, started = self.start_run(base, ["m1"])
            state = self.wait_done(base, started["run_id"], timeout=5)
        self.assertEqual(state["step"], "finished")
        self.assertTrue(any("boom" in line["message"] for line in state["log"]), state["log"])

    def test_a_run_cannot_start_while_an_image_is_being_analyzed(self):
        self.lm.loaded.append("m1")
        self.lm.models["m1"]["delay"] = 0.6
        base = self.start_app()
        answers = {}
        worker = threading.Thread(target=lambda: answers.setdefault("analyze", self.request(base, "/api/analyze?model=m1", PNG)))
        worker.start()
        deadline = time.monotonic() + 2
        while ("chat", "m1") not in self.lm.events and time.monotonic() < deadline:
            time.sleep(0.01)
        status, body = self.start_run(base, ["m3"])
        self.assertEqual(status, 409)
        self.assertIn("being analyzed", body["error"])
        worker.join()
        self.assertEqual(answers["analyze"][0], 200)
        self.assertNotIn(("unload", "m1"), self.lm.events, "the model must not be unloaded under a running request")
        self.assertEqual(self.start_run(base, ["m3"])[0], 200, "free again once the analysis has finished")

    def test_a_large_upload_refused_up_front_gets_its_answer_not_a_reset(self):
        self.lm.models["m1"]["delay"] = 0.5
        base = self.start_app()
        _, started = self.start_run(base, ["m1"])
        big = PNG + b"\0" * (6 * 1024 * 1024)
        status, body = self.request(base, "/api/analyze", big)
        self.assertEqual(status, 409, body)
        self.assertIn("benchmark", body["error"])
        self.wait_done(base, started["run_id"])

    def test_validation(self):
        base = self.start_app()
        self.assertEqual(self.start_run(base, [])[0], 400)                      # nothing selected
        self.assertEqual(self.start_run(base, ["nope"])[0], 400)                # unknown model
        self.assertEqual(self.start_run(base, ["text-llm"])[0], 400)            # not a vision model
        self.assertEqual(self.start_run(base, ["m1"], mode="poem")[0], 400)
        self.assertEqual(self.start_run(base, ["m1"], image=b"not an image")[0], 415)
        self.assertEqual(self.start_run(base, ["m1"], image=b"")[0], 400)
        self.assertEqual(self.lm.events, [])      # nothing happened on the host

    def test_unknown_run_and_latest(self):
        base = self.start_app()
        self.assertEqual(self.request(base, "/api/debug/runs/latest")[0], 404)
        self.assertEqual(self.request(base, "/api/debug/runs/doesnotexist")[0], 404)
        _, started = self.start_run(base, ["m1"])
        self.wait_done(base, started["run_id"])
        status, state = self.request(base, "/api/debug/runs/latest")
        self.assertEqual((status, state["id"]), (200, started["run_id"]))

    def test_diff_endpoint(self):
        base = self.start_app()
        _, started = self.start_run(base, ["m1", "m3"])
        self.wait_done(base, started["run_id"])
        path = "/api/debug/runs/%s/diff?%s" % (started["run_id"], urllib.parse.urlencode({"a": "m1", "b": "m3"}))
        status, body = self.request(base, path)
        self.assertEqual(status, 200)
        self.assertEqual(body["similarity"] < 1.0, True)
        self.assertIn("replace", [op["tag"] for op in body["ops"]])
        bad = "/api/debug/runs/%s/diff?%s" % (started["run_id"], urllib.parse.urlencode({"a": "m1", "b": "zzz"}))
        self.assertEqual(self.request(base, bad)[0], 404)

    def test_progress_is_visible_while_running(self):
        self.lm.models["m1"]["delay"] = 0.4
        base = self.start_app()
        _, started = self.start_run(base, ["m1", "m3"])
        time.sleep(0.15)
        _, state = self.request(base, "/api/debug/runs/" + started["run_id"])
        self.assertEqual(state["status"], "running")
        self.assertEqual((state["total"], state["current_model"]), (2, "m1"))
        self.assertIn(state["step"], ("unloading", "loading", "inferring"))
        self.assertIsNone(state["analysis"])
        self.wait_done(base, started["run_id"])


class SavedRuns(BenchmarkTestCase):
    """Finished runs are written to DATA_DIR as JSON and can be fetched again, also after a restart."""

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_dir = self.tmp.name

    def finished_run(self, base, models=("m1", "m3")):
        _, started = self.start_run(base, list(models))
        return self.wait_done(base, started["run_id"])

    def test_finished_run_is_written_as_json_file(self):
        base = self.start_app(data_dir=self.data_dir)
        state = self.finished_run(base)
        path = os.path.join(self.data_dir, "runs", state["id"] + ".json")
        with open(path, encoding="utf-8") as handle:
            saved = json.load(handle)
        self.assertEqual(saved["id"], state["id"])
        self.assertEqual([r["model"] for r in saved["results"]], ["m1", "m3"])
        self.assertEqual(saved["status"], "done")

    def test_the_uploaded_image_is_never_written_to_disk(self):
        image = PNG + b"UNIQUE-IMAGE-MARKER" * 50
        base = self.start_app(data_dir=self.data_dir)
        _, started = self.start_run(base, ["m1", "m3"], image=image)
        state = self.wait_done(base, started["run_id"])
        needles = (b"UNIQUE-IMAGE-MARKER", base64.b64encode(image)[:40], base64.b64encode(image[1:])[:40])
        written = []
        for folder, _, files in os.walk(self.data_dir):
            for name in files:
                with open(os.path.join(folder, name), "rb") as handle:
                    written.append(handle.read())
        self.assertEqual(len(written), 1)                  # only the run's JSON
        for data in written:
            for needle in needles:
                self.assertNotIn(needle, data)
        run = self.apps[-1].RequestHandlerClass.runs.get(state["id"])
        self.assertIsNone(run.image)                       # and the finished run drops it from memory

    def test_saved_list_is_newest_first_with_summary(self):
        base = self.start_app(data_dir=self.data_dir)
        first = self.finished_run(base, ("m1",))
        second = self.finished_run(base, ("m1", "m4"))
        status, body = self.request(base, "/api/debug/saved")
        self.assertEqual(status, 200)
        self.assertTrue(body["enabled"])
        self.assertEqual([r["id"] for r in body["runs"]], [second["id"], first["id"]])
        entry = body["runs"][0]
        self.assertEqual((entry["status"], entry["mode"], entry["total"], entry["ok"]), ("done", "ocr", 2, 2))
        self.assertNotIn("results", entry)

    def test_saved_run_survives_a_restart_including_diff_and_latest(self):
        state = self.finished_run(self.start_app(data_dir=self.data_dir))
        again = self.start_app(data_dir=self.data_dir)      # a new server: empty memory
        status, loaded = self.request(again, "/api/debug/runs/" + state["id"])
        self.assertEqual(status, 200)
        self.assertEqual(loaded["results"], state["results"])
        self.assertEqual(loaded["analysis"], state["analysis"])
        self.assertEqual(self.request(again, "/api/debug/runs/latest")[1]["id"], state["id"])
        path = "/api/debug/runs/%s/diff?%s" % (state["id"], urllib.parse.urlencode({"a": "m1", "b": "m3"}))
        status, diff = self.request(again, path)
        self.assertEqual(status, 200)
        self.assertIn("replace", [op["tag"] for op in diff["ops"]])

    def test_without_data_dir_nothing_is_written_and_list_is_empty(self):
        base = self.start_app()
        self.finished_run(base, ("m1",))
        status, body = self.request(base, "/api/debug/saved")
        self.assertEqual((status, body), (200, {"enabled": False, "runs": []}))

    def test_corrupt_and_foreign_files_are_skipped(self):
        runs_dir = os.path.join(self.data_dir, "runs")
        os.makedirs(runs_dir)
        for name, content in (("deadbeef0001.json", "{not json"), ("notes.txt", "x"), ("deadbeef0002.json", "[1]")):
            with open(os.path.join(runs_dir, name), "w") as handle:
                handle.write(content)
        base = self.start_app(data_dir=self.data_dir)
        self.assertEqual(self.request(base, "/api/debug/saved")[1]["runs"], [])
        self.assertEqual(self.request(base, "/api/debug/runs/deadbeef0001")[0], 404)
        self.assertEqual(self.request(base, "/api/debug/runs/latest")[0], 404)

    def test_files_with_wrong_field_types_are_skipped_not_a_500(self):
        runs_dir = os.path.join(self.data_dir, "runs")
        os.makedirs(runs_dir)
        bad = {"aaaaaaaaaaa1": {"id": "aaaaaaaaaaa1", "results": ["not a dict"], "started_at": 1},
               "aaaaaaaaaaa2": {"id": "aaaaaaaaaaa2", "results": [], "started_at": "yesterday"},
               "aaaaaaaaaaa3": {"id": "aaaaaaaaaaa3", "results": [{"model": "m1", "status": "ok", "text": 5}]}}
        for run_id, data in bad.items():
            with open(os.path.join(runs_dir, run_id + ".json"), "w") as handle:
                json.dump(data, handle)
        base = self.start_app(data_dir=self.data_dir)
        self.assertEqual(self.request(base, "/api/debug/saved"), (200, {"enabled": True, "runs": []}))
        for run_id in bad:
            self.assertEqual(self.request(base, "/api/debug/runs/" + run_id)[0], 404)

    def test_older_file_without_tree_gets_its_analysis_rebuilt(self):
        state = self.finished_run(self.start_app(data_dir=self.data_dir))
        path = os.path.join(self.data_dir, "runs", state["id"] + ".json")
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        for key in ("tree", "agreement"):
            del data["analysis"][key]           # what an export from version 0.1.7 looks like
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        _, loaded = self.request(self.start_app(data_dir=self.data_dir), "/api/debug/runs/" + state["id"])
        self.assertEqual(loaded["analysis"], state["analysis"])

    def test_a_hand_made_file_gets_every_field_the_page_reads(self):
        runs_dir = os.path.join(self.data_dir, "runs")
        os.makedirs(runs_dir)
        with open(os.path.join(runs_dir, "0123456789ab.json"), "w", encoding="utf-8") as handle:
            json.dump({"id": "0123456789ab", "status": "done", "results": [
                {"model": "m1", "status": "ok", "text": "Rechnung 1"}, {"model": "m2", "status": "ok", "text": "Rechnung 1"}],
                "analysis": {"tree": None, "agreement": {}}}, handle)
        status, loaded = self.request(self.start_app(data_dir=self.data_dir), "/api/debug/runs/0123456789ab")
        self.assertEqual(status, 200, loaded)
        self.assertEqual(loaded["log"], [])
        self.assertEqual(loaded["restore"], {"requested": False, "models": [], "loaded": [], "failed": []})
        self.assertEqual(loaded["analysis"]["matrix"]["models"], ["m1", "m2"])
        self.assertEqual(loaded["mode"], "ocr")

    def test_failed_write_leaves_no_temp_file(self):
        runs_dir = os.path.join(self.data_dir, "runs")
        os.makedirs(os.path.join(runs_dir, "aaaaaaaaaaaa.json"))      # a directory where the file should go
        store = benchmark.RunStore(self.data_dir)
        store.save({"id": "aaaaaaaaaaaa", "results": []})
        self.assertEqual(sorted(os.listdir(runs_dir)), ["aaaaaaaaaaaa.json"])

    def test_run_id_cannot_escape_the_data_dir(self):
        with open(os.path.join(self.tmp.name, "secret.json"), "w") as handle:
            json.dump({"id": "secret", "results": []}, handle)
        base = self.start_app(data_dir=os.path.join(self.data_dir, "inner"))
        for run_id in ("..%2Fsecret", "..%2F..%2Fsecret", "%2E%2E%2Fsecret"):
            self.assertEqual(self.request(base, "/api/debug/runs/" + run_id)[0], 404, run_id)

    def test_unwritable_data_dir_does_not_fail_the_run(self):
        blocker = os.path.join(self.data_dir, "file")
        with open(blocker, "w") as handle:
            handle.write("x")
        base = self.start_app(data_dir=os.path.join(blocker, "sub"))   # a path below a regular file
        state = self.finished_run(base, ("m1",))
        self.assertEqual(state["status"], "done")


class ErrorMessages(unittest.TestCase):
    def test_nested_error_object_uses_its_message(self):
        self.assertEqual(backends.error_message({"error": {"type": "x", "message": "boom"}}), "boom")
        self.assertEqual(backends.error_message({"error": "plain"}), "plain")
        self.assertIsNone(backends.error_message({"status": "loaded"}))


if __name__ == "__main__":
    unittest.main()
