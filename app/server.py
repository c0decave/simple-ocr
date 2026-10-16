"""Minimal web app: send an image to a local vision model (LM Studio, Ollama or vLLM), get its text or a description.

Standard library only. Configuration comes from environment variables, see Config.from_env().
"""
import json
import os
import re
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app import backends, batch, benchmark, compare, report
from app.backends import ApiError

INDEX_PATH = Path(__file__).with_name("index.html")
DEBUG_PATH = Path(__file__).with_name("debug.html")
BATCH_PATH = Path(__file__).with_name("batch.html")
VERSION_PATH = Path(__file__).parent.parent / "VERSION"   # copied next to app/ in the image
VERSION_PLACEHOLDER = "__APP_VERSION__"


def read_version(path):
    """The x.y.z from the VERSION file, or "unknown" when it is missing or not a version."""
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"
    return text if re.fullmatch(r"\d+\.\d+\.\d+", text) else "unknown"


APP_VERSION = read_version(VERSION_PATH)

PROMPT_MAX_CHARS = 4000   # a custom prompt travels in the query string
# Optional parts of the pages, kept only when their switch is on: <!--NAME--> ... <!--/NAME-->
PAGE_PARTS = {"DEBUG": "debug_page", "PROMPT": "custom_prompt", "BATCH": "batch_page"}
REPORTS = {"md": ("text/markdown; charset=utf-8", report.markdown), "html": ("text/html; charset=utf-8", report.html),
           "pdf": ("application/pdf", report.pdf)}

PROMPTS = {
    "ocr": (
        "Extract all text from this image exactly as it appears. "
        "Keep the reading order and the line breaks. Do not translate, summarize or comment. "
        "Output only the extracted text."
    ),
    "describe": (
        "Describe what you see in this image: the main subjects, the setting, notable details "
        "and any visible text. Be concise and factual. Reply in {language}."
    ),
}

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
# IBM Granite Vision wraps its whole answer in <doc>...</doc>; only a wrapper around everything is removed.
DOC_WRAPPER = re.compile(r"\A\s*<doc>(.*)</doc>\s*\Z", re.DOTALL)


@dataclass
class Config:
    backend: str = "lmstudio"        # lmstudio, ollama or vllm, see app.backends.BACKENDS
    backend_url: str = "http://localhost:1234"
    api_key: str = field(default="", repr=False)  # never printed, never sent to the browser
    default_model: str = ""
    default_language: str = "German"
    max_upload_bytes: int = 20 * 1024 * 1024
    timeout: float = 300.0
    data_dir: str = ""   # finished benchmark runs are saved below it; empty = not saved
    tls_verify: bool = True   # False accepts any https certificate (self-signed), see <PREFIX>_TLS_VERIFY
    # hard-coded in app/backends.py, never read from the environment; looked up when a Config is made
    allow_loading: bool = field(default_factory=lambda: backends.MODEL_LOADING_ALLOWED)
    custom_prompt: bool = True   # CUSTOM_PROMPT: the prompt may be changed per request
    debug_page: bool = True      # DEBUG_PAGE: /debug and every /api/debug route
    batch_page: bool = True      # BATCH_PAGE: /batch and every /api/batch route
    max_batch_bytes: int = 200 * 1024 * 1024   # MAX_BATCH_MB: one batch upload

    @property
    def backend_name(self):
        return backends.BACKENDS[self.backend][0]

    @classmethod
    def from_env(cls):
        env = os.environ
        backend = env.get("BACKEND", "").strip().lower() or "lmstudio"
        if backend not in backends.BACKENDS:
            raise ValueError("BACKEND must be one of: %s, got %r" % (", ".join(backends.BACKENDS), backend))
        _, prefix, port, _ = backends.BACKENDS[backend]
        return cls(
            backend=backend,
            backend_url=_backend_url(env, prefix, port, url_in_host=backend == "ollama"),
            api_key=env.get(prefix + "_API_KEY", "").strip(),
            default_model=env.get("DEFAULT_MODEL", ""),
            default_language=env.get("DESCRIBE_LANGUAGE", cls.default_language),
            max_upload_bytes=int(float(env.get("MAX_UPLOAD_MB", "20")) * 1024 * 1024),
            timeout=float(env.get("UPSTREAM_TIMEOUT", "300")),
            data_dir=env.get("DATA_DIR", "").strip(),
            tls_verify=_flag(env, prefix + "_TLS_VERIFY", default=True),
            custom_prompt=_flag(env, "CUSTOM_PROMPT", default=True),
            debug_page=_flag(env, "DEBUG_PAGE", default=True),
            batch_page=_flag(env, "BATCH_PAGE", default=True),
            max_batch_bytes=int(float(env.get("MAX_BATCH_MB", "200")) * 1024 * 1024),
        )


def _flag(env, name, default):
    value = env.get(name, "").strip().lower()
    if not value:
        return default
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError("%s must be true or false, got %r" % (name, value))


def _backend_url(env, prefix, default_port, url_in_host=False):
    """<PREFIX>_URL wins; otherwise http://<PREFIX>_HOST:<PREFIX>_PORT.

    HOST may carry the port ("host:1234"). A full URL in HOST is only accepted for Ollama, whose own
    OLLAMA_HOST works that way; the others have <PREFIX>_URL for it.
    """
    url = env.get(prefix + "_URL", "").strip()
    if url:
        if not url.lower().startswith(("http://", "https://")):
            raise ValueError("%s_URL must start with http:// or https://" % prefix)
        return url
    host = env.get(prefix + "_HOST", "").strip() or "localhost"
    port = env.get(prefix + "_PORT", "").strip() or str(default_port)
    if "://" in host:
        if url_in_host:
            return host
        raise ValueError("%s_HOST must be a host name or IP address; use %s_URL for a full URL" % (prefix, prefix))
    if host.count(":") == 1 and not host.startswith("["):
        host, port = host.split(":")   # host:port
    elif host.startswith("[") and "]:" in host:
        host, port = host[1:].split("]:")   # [IPv6]:port
    if not port.isdigit():
        raise ValueError("%s_PORT must be a number, got %r" % (prefix, port))
    if ":" in host and not host.startswith("["):
        host = "[%s]" % host  # IPv6 literal
    return "http://%s:%s" % (host, port)


def parse_env_file(path):
    """Parse a simple KEY=VALUE file. Missing file gives {}. Errors name the line, never its content."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    values = {}
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError("%s: line %d is not KEY=VALUE" % (path, number))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_env_file(path):
    """Put the file's values into os.environ without overriding variables that are already set."""
    for key, value in parse_env_file(path).items():
        os.environ.setdefault(key, value)



def sniff_image_type(data):
    """Return the MIME type from the file's magic bytes, or None if it is not a supported image."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def fetch_models(config):
    """Vision models (loaded first) and everything currently loaded on the backend, any type."""
    models, loaded_now = backends.list_models(config)
    models.sort(key=lambda m: not m["loaded"])  # stable: loaded first, original order otherwise
    return {"models": models, "loaded_now": loaded_now}


def usable(config, models):
    """The models the interface may use: all of them, or only the loaded ones when loading is not allowed."""
    return models if config.allow_loading else [m for m in models if m["loaded"] is True]


def resolve_model(config, models, name):
    """(listed id, None) for a vision model name, else (None, why not).

    Ollama names are name:tag; a name without its tag is completed when exactly one tag matches.
    """
    ids = [m["id"] for m in models]
    if name in ids:
        return name, None
    tagged = [i for i in ids if i.startswith(name + ":")] if ":" not in name else []
    if len(tagged) == 1:
        return tagged[0], None
    if tagged:
        return None, "%s matches several models on %s: %s; name one with its tag" % (
            name, config.backend_name, ", ".join(tagged))
    return None, "%s is not a vision model on %s" % (name, config.backend_name)


def default_notice(config, models):
    """Why DEFAULT_MODEL is not used, for the page; None when it is used or not set."""
    if not config.default_model:
        return None
    resolved, problem = resolve_model(config, models, config.default_model)
    if problem:
        return "DEFAULT_MODEL " + problem
    if resolved not in [m["id"] for m in usable(config, models)]:
        return "DEFAULT_MODEL %s is not loaded on %s, and loading models is disabled" % (resolved, config.backend_name)
    return None


def pick_default(config, models):
    """Configured model, else the first loaded one, else the smallest (cheapest to load), else the first."""
    models = usable(config, models)
    if config.default_model:
        resolved, _ = resolve_model(config, models, config.default_model)
        if resolved:
            return resolved
    for model in models:
        if model["loaded"]:
            return model["id"]
    sized = [m for m in models if m.get("size_bytes")]
    if sized:
        return min(sized, key=lambda m: m["size_bytes"])["id"]
    return models[0]["id"] if models else ""


def default_prompts(config):
    return {mode: text.format(language=config.default_language) for mode, text in PROMPTS.items()}


def check_task(config, mode, prompt):
    """Mode and custom prompt of a request; returns the prompt, stripped ("" = the default prompt)."""
    if mode not in PROMPTS:
        raise ApiError(400, "Unknown mode %r, expected one of: %s" % (mode, ", ".join(PROMPTS)))
    prompt = (prompt or "").strip()
    if prompt and not config.custom_prompt:
        raise ApiError(403, "Custom prompts are disabled on this server")
    if len(prompt) > PROMPT_MAX_CHARS:
        raise ApiError(400, "The prompt is too long: %d characters, at most %d" % (len(prompt), PROMPT_MAX_CHARS))
    return prompt


def choose_model(config, model):
    """The listed vision model to use for a request ("" = the default one), or an ApiError."""
    models = fetch_models(config)["models"]
    if model:
        # only a listed vision model reaches the server: a text model or a name without its tag
        # would come back as a bare "model not found"
        resolved, problem = resolve_model(config, models, model)
        if problem:
            raise ApiError(400, problem)
        model = resolved
        if not any(m["id"] == model for m in usable(config, models)):
            raise ApiError(403, "%s is not loaded on %s, and loading models is disabled" % (model, config.backend_name))
    else:
        model = pick_default(config, models)
        if not model:
            if not config.allow_loading:
                raise ApiError(503, "No vision model is loaded on %s, and loading models is disabled" % config.backend_name)
            raise ApiError(503, "No vision model is available on %s" % config.backend_name)
    return model


def analyze(config, image, mode, model, prompt=None):
    """prompt: a custom prompt sent as it is; None or blank means the default prompt of the mode."""
    mime = sniff_image_type(image)
    if mime is None:
        raise ApiError(415, "Unsupported file: only PNG, JPEG and WebP images are accepted")
    prompt = check_task(config, mode, prompt)
    model = choose_model(config, model)
    prompt = prompt or default_prompts(config)[mode]
    started = time.monotonic()
    content = backends.chat(config, model, prompt, mime, image)
    text = THINK_BLOCK.sub("", content or "").strip()
    wrapped = DOC_WRAPPER.match(text)
    if wrapped:
        text = wrapped.group(1).strip()
    if not text:
        raise ApiError(502, "The model returned no text")
    return {"text": text, "mode": mode, "model": model, "seconds": round(time.monotonic() - started, 1)}


class Handler(BaseHTTPRequestHandler):
    config: Config           # set by make_server()
    runs: "benchmark.Runs"   # set by make_server()
    batches: "batch.Batches"   # set by make_server()
    timeout = 60    # seconds a client may stall while sending its request

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, payload):
        self._send(status, json.dumps(payload).encode(), "application/json; charset=utf-8")

    def _fail(self, error):
        self._json(error.status, {"error": error.message})

    def _guarded(self, action):
        """Run a route. Every outcome becomes a JSON answer, so the browser never sees a dropped connection."""
        try:
            action()
        except ApiError as error:
            self._reply_error(error)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client hung up, e.g. a newer request replaced this one
        except Exception as error:  # noqa: BLE001 - last line of defence for the route table
            traceback.print_exc()
            self._reply_error(ApiError(500, "Internal error: %s: %s" % (type(error).__name__, error)))

    def _reply_error(self, error):
        try:
            if self.command == "POST" and not getattr(self, "_body_read", True):
                # refused before the upload was read (409, 403, ...): read it first, or the client sees a
                # connection reset instead of the answer
                try:
                    length = int(self.headers.get("Content-Length", ""))
                except ValueError:
                    length = 0
                    self.close_connection = True
                self._drain(length, self.config.max_upload_bytes * 4)
            self._fail(error)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        self._guarded(self._route_get)

    def do_POST(self):
        self._body_read = False
        self._guarded(self._route_post)

    def _page(self, path):
        html = path.read_text(encoding="utf-8").replace(VERSION_PLACEHOLDER, APP_VERSION)
        for name, switch in PAGE_PARTS.items():
            part = re.compile(r"<!--%s-->(.*?)<!--/%s-->" % (name, name), re.DOTALL)
            html = part.sub(r"\1" if getattr(self.config, switch) else "", html)
        self._send(200, html.encode(), "text/html; charset=utf-8")

    def _check_switches(self, path):
        """The .env switches are enforced here, for every route of the page, not only by hiding the button."""
        if (path == "/debug" or path.startswith("/api/debug/")) and not self.config.debug_page:
            raise ApiError(403, "The debug page is disabled on this server")
        if path in ("/batch", "/api/batch") or path.startswith("/api/batch/"):
            if not self.config.batch_page:
                raise ApiError(403, "The batch page is disabled on this server")

    def _route_get(self):
        path = urlparse(self.path).path
        self._check_switches(path)
        if path == "/":
            self._page(INDEX_PATH)
        elif path == "/api/settings":
            self._json(200, {"custom_prompt": self.config.custom_prompt, "debug_page": self.config.debug_page,
                             "batch_page": self.config.batch_page,
                             "prompts": default_prompts(self.config), "prompt_max_chars": PROMPT_MAX_CHARS})
        elif path == "/api/models":
            listing = fetch_models(self.config)
            listing["default"] = pick_default(self.config, listing["models"])
            notice = default_notice(self.config, listing["models"])
            if notice:
                listing["notice"] = notice
            listing["backend"] = backends.describe(self.config)
            self._json(200, listing)
        elif path == "/api/analyze":
            raise ApiError(405, "Use POST with the image as request body")
        elif path == "/debug":
            self._page(DEBUG_PATH)
        elif path == "/api/debug/saved":
            self._json(200, self.runs.saved())
        elif path.startswith("/api/debug/runs/"):
            self._route_debug_get(path[len("/api/debug/runs/"):], parse_qs(urlparse(self.path).query))
        elif path == "/batch":
            self._page(BATCH_PATH)
        elif path == "/api/batch":
            self._json(200, {"batches": self.batches.listing()})
        elif path.startswith("/api/batch/"):
            self._route_batch_get(path[len("/api/batch/"):])
        else:
            raise ApiError(404, "Not found")

    def _find_run(self, run_id):
        run = self.runs.get(run_id)
        if run is None:
            raise ApiError(404, "No such run")
        return run

    def _route_debug_get(self, rest, query):
        run_id, _, action = rest.partition("/")
        run = self._find_run(run_id)
        if action == "":
            self._json(200, run.snapshot())
        elif action == "diff":
            a, b = query.get("a", [""])[0], query.get("b", [""])[0]
            text_a, text_b = run.result_text(a), run.result_text(b)
            if text_a is None or text_b is None:
                raise ApiError(404, "No successful result for %r and %r in this run" % (a, b))
            self._json(200, {"a": a, "b": b, "similarity": round(compare.similarity(text_a, text_b), 4),
                             "ops": compare.diff(text_a, text_b)})
        else:
            raise ApiError(404, "Not found")

    def _find_batch(self, job_id):
        job = self.batches.get(job_id)
        if job is None:
            raise ApiError(404, "No such batch")
        return job

    def _route_batch_get(self, rest):
        job_id, _, action = rest.partition("/")
        job = self._find_batch(job_id)
        if action == "":
            self._json(200, job.snapshot())
        elif action.startswith("report.") and action[len("report."):] in REPORTS:
            ext = action[len("report."):]
            content_type, make = REPORTS[ext]
            body = make(job)
            body = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Disposition", 'attachment; filename="simple-ocr-batch-%s.%s"' % (job.id, ext))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            raise ApiError(404, "Not found")

    def _start_batch(self, query):
        mode = query.get("mode", ["ocr"])[0]
        prompt = check_task(self.config, mode, query.get("prompt", [""])[0])
        embed = query.get("images", [""])[0] in ("1", "true")
        content_type = self.headers.get("Content-Type", "")
        body = self._read_body(self.config.max_batch_bytes)
        files = batch.parse_upload(content_type, body)
        del body
        images, skipped = batch.collect(files, self.config.max_upload_bytes, self.config.max_batch_bytes)
        del files
        if not images:
            raise ApiError(400, "No image in the upload (PNG, JPEG or WebP, also inside ZIP archives)"
                           + "".join("; %s: %s" % entry for entry in skipped[:5]))
        model = choose_model(self.config, query.get("model", [""])[0])
        job = self.batches.start(self.config, mode, model, prompt, embed, images, skipped, self.runs)
        self._json(200, {"id": job.id, "model": model, "total": len(job.items),
                         "skipped": [{"name": name, "reason": reason} for name, reason in skipped]})

    def _route_post(self):
        parsed = urlparse(self.path)
        self._check_switches(parsed.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/api/analyze":
            mode = query.get("mode", ["ocr"])[0]
            model = query.get("model", [""])[0]
            prompt = query.get("prompt", [""])[0]
            self.runs.begin_analysis()
            try:
                image = self._read_body()
                result = analyze(self.config, image, mode, model, prompt)
            finally:
                self.runs.end_analysis()
            self._json(200, result)
        elif parsed.path == "/api/debug/runs":
            self._start_run(query)
        elif parsed.path == "/api/batch":
            self._start_batch(query)
        elif parsed.path.startswith("/api/batch/") and parsed.path.endswith("/cancel"):
            self._find_batch(parsed.path.split("/")[3]).cancel()
            self._json(200, {"cancelling": True})
        elif parsed.path.startswith("/api/debug/runs/") and parsed.path.endswith("/cancel"):
            self._find_run(parsed.path.split("/")[4]).cancel()
            self._json(200, {"cancelling": True})
        else:
            raise ApiError(404, "Not found")

    def _start_run(self, query):
        mode = query.get("mode", ["ocr"])[0]
        wanted = [m for m in query.get("model", []) if m]
        every = query.get("all", [""])[0] in ("1", "true")
        image = self._read_body()
        if sniff_image_type(image) is None:
            raise ApiError(415, "Unsupported file: only PNG, JPEG and WebP images are accepted")
        if mode not in PROMPTS:
            raise ApiError(400, "Unknown mode %r, expected one of: %s" % (mode, ", ".join(PROMPTS)))
        if not wanted and not every:
            raise ApiError(400, "No model selected")
        listing = fetch_models(self.config)
        available = {m["id"]: m for m in listing["models"]}
        allowed = {m["id"] for m in usable(self.config, listing["models"])}
        if every:
            chosen = [m for m in available.values() if m["id"] in allowed]
        else:
            unknown = [m for m in dict.fromkeys(wanted) if m not in available]
            if unknown:
                raise ApiError(400, "Not a vision model on %s: %s" % (self.config.backend_name, ", ".join(unknown)))
            refused = [m for m in dict.fromkeys(wanted) if m not in allowed]
            if refused:
                raise ApiError(403, "Not loaded on %s, and loading models is disabled: %s"
                               % (self.config.backend_name, ", ".join(refused)))
            chosen = [available[m] for m in dict.fromkeys(wanted)]
        if not chosen:
            raise ApiError(400, "%s lists no usable vision model" % self.config.backend_name)
        # vLLM cannot load or unload, and with loading disabled nothing is unloaded: nothing to restore either
        restore = query.get("restore", [""])[0] in ("1", "true") and backends.manages_models(self.config)
        previous = [m["id"] for m in listing["loaded_now"]] if restore else []
        run = self.runs.start(self.config, mode, image, chosen, restore, previous)
        self._json(200, {"run_id": run.id, "models": [m["id"] for m in chosen]})

    def _read_body(self, limit=None):
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise ApiError(411, "Content-Length header is required")
        limit = limit or self.config.max_upload_bytes
        if length <= 0:
            raise ApiError(400, "Empty request body: send the image as the request body")
        if length > limit:
            self._drain(length, limit * 4)
            raise ApiError(413, "The upload is larger than the %.0f MB limit" % (limit / 1024 / 1024))
        body = self.rfile.read(length)
        self._body_read = True
        if len(body) != length:
            raise ApiError(400, "Incomplete upload: expected %d bytes, received %d" % (length, len(body)))
        return body

    def _drain(self, length, cap):
        """Read and discard the body so the browser sees the 413 instead of a connection reset."""
        self._body_read = True
        remaining = min(length, cap)
        while remaining > 0:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
        if length > cap:
            self.close_connection = True

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def make_server(config, host, port):
    handler = type("BoundHandler", (Handler,), {"config": config, "runs": benchmark.Runs(config.data_dir),
                                                "batches": batch.Batches()})
    return ThreadingHTTPServer((host, port), handler)


def startup_line(config, host, port):
    line = "simple-ocr listening on http://%s:%d, %s: %s" % (host, port, config.backend_name, config.backend_url)
    if not config.tls_verify:
        line += " (TLS certificate is NOT verified)"
    return line


def main():
    load_env_file(os.environ.get("ENV_FILE", ".env"))
    config = Config.from_env()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    httpd = make_server(config, host, port)
    # As PID 1 in a container Python ignores SIGTERM by default; treat it like Ctrl+C.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    print(startup_line(config, host, port), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
