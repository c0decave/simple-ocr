"""Tests for the Ollama and vLLM backends, each against a stateful fake built from its API docs.

Ollama: docs/api.md (/api/tags, /api/show capabilities, /api/ps, /api/generate keep_alive, /api/chat images).
vLLM: OpenAI-compatible server (/v1/models, /v1/chat/completions with data: URLs, --api-key as Bearer).
llama.cpp: tools/server/README.md (/v1/models with architecture.input_modalities; router mode: GET /models
with status, POST /models/load and /models/unload).
"""
import atexit
import base64
import json
import os
import shutil
import ssl
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from app import backends, benchmark, server

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
GB = 1000 ** 3


_tls_dir = None


def self_signed_pair():
    """A throwaway self-signed certificate and key, made once per test run (none is kept in the repo)."""
    global _tls_dir
    if _tls_dir is None:
        if not shutil.which("openssl"):
            raise unittest.SkipTest("openssl is needed to make a test certificate")
        _tls_dir = tempfile.mkdtemp(prefix="simple-ocr-tls-")
        atexit.register(shutil.rmtree, _tls_dir, True)
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2", "-subj", "/CN=localhost",
                        "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
                        "-keyout", os.path.join(_tls_dir, "test.key"), "-out", os.path.join(_tls_dir, "test.crt")],
                       check=True, capture_output=True)
    return os.path.join(_tls_dir, "test.crt"), os.path.join(_tls_dir, "test.key")


class FakeHttp:
    """Small JSON server; subclasses implement route(method, path, body) -> (status, payload)."""

    tls = False   # True: HTTPS with a self-signed certificate from self_signed_pair()

    def __init__(self):
        self.requests = []      # (method, path, body, authorization)
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self, method):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"null") if length else None
                with outer.lock:
                    outer.requests.append((method, self.path, body, self.headers.get("Authorization")))
                status, payload = outer.route(method, self.path, body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        if self.tls:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(*self_signed_pair())
            self.httpd.socket = context.wrap_socket(self.httpd.socket, server_side=True)
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        self.url = "%s://127.0.0.1:%d" % ("https" if self.tls else "http", self.httpd.server_address[1])

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def paths(self, method=None):
        return [(m, p) for m, p, _, _ in self.requests if method is None or m == method]


class FakeOllama(FakeHttp):
    def __init__(self, models, loaded=()):
        self.models = models     # name -> dict(caps, families, size, params, quant, text, load_ns, chat_error)
        self.loaded = list(loaded)
        self.events = []         # ("load", name, loaded_before) / ("unload", name) / ("chat", name)
        super().__init__()

    def _details(self, spec):
        return {"format": "gguf", "family": "x", "families": spec.get("families", ["x"]),
                "parameter_size": spec.get("params", "2B"), "quantization_level": spec.get("quant", "Q4_K_M")}

    def route(self, method, path, body):
        with self.lock:
            if (method, path) == ("GET", "/api/tags"):
                return 200, {"models": [{"name": n, "model": n, "size": s.get("size", GB), "digest": "d-" + n,
                                         "details": self._details(s)} for n, s in self.models.items()]}
            if (method, path) == ("GET", "/api/ps"):
                return 200, {"models": [{"name": n, "model": n, "size": 2 * self.models[n].get("size", GB),
                                         "size_vram": 0, "digest": "d-" + n, "details": self._details(self.models[n])}
                                        for n in self.loaded]}
            name = (body or {}).get("model")
            spec = self.models.get(name)
            if path in ("/api/show", "/api/generate", "/api/chat") and spec is None:
                return 404, {"error": "model '%s' not found" % name}
            if (method, path) == ("POST", "/api/show"):
                info = {"details": self._details(spec), "model_info": {}}
                if "caps" in spec:
                    info["capabilities"] = spec["caps"]
                return 200, info
            if (method, path) == ("POST", "/api/generate"):
                if body.get("keep_alive") == 0:
                    if name in self.loaded:
                        self.loaded.remove(name)
                    self.events.append(("unload", name))
                    return 200, {"model": name, "response": "", "done": True, "done_reason": "unload"}
                self.events.append(("load", name, list(self.loaded)))
                if name not in self.loaded:
                    self.loaded.append(name)
                reply = {"model": name, "response": "", "done": True}
                if "load_ns" in spec:
                    reply["load_duration"] = spec["load_ns"]
                return 200, reply
            if (method, path) == ("POST", "/api/chat"):
                self.events.append(("chat", name))
                if spec.get("chat_error"):
                    return 500, {"error": spec["chat_error"]}
                if name not in self.loaded:
                    self.loaded.append(name)
                return 200, {"model": name, "message": {"role": "assistant", "content": spec.get("text", "")},
                             "done": True, "done_reason": "stop"}
        return 404, {"error": "no route"}


class FakeVllm(FakeHttp):
    def __init__(self, models, key=None):
        self.models, self.key = models, key   # id -> text
        super().__init__()

    def route(self, method, path, body):
        auth = self.requests[-1][3]
        if self.key and auth != "Bearer " + self.key:
            return 401, {"error": "Unauthorized"}
        if (method, path) == ("GET", "/v1/models"):
            return 200, {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "vllm", "root": m,
                                                     "max_model_len": 32768} for m in self.models]}
        if (method, path) == ("POST", "/v1/chat/completions"):
            if body["model"] not in self.models:
                return 404, {"object": "error", "message": "The model `%s` does not exist." % body["model"],
                             "type": "NotFoundError", "code": 404}
            return 200, {"choices": [{"message": {"role": "assistant", "content": self.models[body["model"]]}}]}
        return 404, {"error": "no route"}


class FakeLlamaCpp(FakeHttp):
    """router=False: one fixed model. router=True: models with status; load and unload finish asynchronously."""

    def __init__(self, models, router, loaded=(), load_delay=0.05):
        self.models, self.router, self.load_delay = models, router, load_delay   # id -> dict(modalities, text, fail)
        self.status = {m: ("loaded" if m in loaded or not router else "unloaded") for m in models}
        self.events = []
        super().__init__()

    def _entry(self, model_id):
        spec = self.models[model_id]
        entry = {"id": model_id, "object": "model", "owned_by": "llamacpp"}
        if "modalities" in spec:
            entry["architecture"] = {"input_modalities": spec["modalities"], "output_modalities": ["text"]}
        if "meta" in spec:
            entry["meta"] = spec["meta"]
        if self.router:
            entry["status"] = {"value": self.status[model_id], "args": ["llama-server", "-ctx", "4096"]}
            if self.status[model_id] == "unloaded" and spec.get("failed"):
                entry["status"].update(failed=True, exit_code=1)
        return entry

    def _finish_later(self, model_id, value):
        def done():
            time.sleep(self.load_delay)
            with self.lock:
                if value == "loaded" and self.models[model_id].get("fail"):
                    self.status[model_id] = "unloaded"
                    self.models[model_id]["failed"] = True
                else:
                    self.status[model_id] = value
        threading.Thread(target=done, daemon=True).start()

    def route(self, method, path, body):
        with self.lock:
            if method == "GET" and path in ("/models", "/v1/models"):
                # measured on llama-server b11610: without router mode /models answers like /v1/models
                return 200, {"object": "list", "data": [self._entry(m) for m in self.models]}
            if self.router and (method, path) == ("POST", "/models/load"):
                self.events.append(("load", body["model"], sorted(m for m, v in self.status.items() if v == "loaded")))
                self.models[body["model"]].pop("failed", None)
                self.status[body["model"]] = "loading"
                self._finish_later(body["model"], "loaded")
                return 200, {"success": True}
            if self.router and (method, path) == ("POST", "/models/unload"):
                self.events.append(("unload", body["model"]))
                self._finish_later(body["model"], "unloaded")
                return 200, {"success": True}
            if (method, path) == ("POST", "/v1/chat/completions"):
                self.events.append(("chat", body["model"]))
                return 200, {"choices": [{"message": {"role": "assistant", "content": self.models[body["model"]]["text"]}}]}
        return 404, {"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}}


class AppMixin:
    def start_app(self, **config):
        # loading allowed unless a test asks otherwise; MODEL_LOADING_ALLOWED=false is ModelLoadingDisabled
        app = server.make_server(server.Config(**dict({"timeout": 5, "allow_loading": True}, **config)), "127.0.0.1", 0)
        threading.Thread(target=app.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        self.addCleanup(app.server_close)
        self.addCleanup(app.shutdown)
        return "http://127.0.0.1:%d" % app.server_address[1]

    def request(self, base, path, body=None):
        req = urllib.request.Request(base + path, data=body, method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def run_benchmark(self, base, models, restore=False):
        query = [("mode", "ocr")] + [("model", m) for m in models] + ([("restore", "1")] if restore else [])
        status, started = self.request(base, "/api/debug/runs?" + urllib.parse.urlencode(query), PNG)
        self.assertEqual(status, 200, started)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            _, state = self.request(base, "/api/debug/runs/" + started["run_id"])
            if state["status"] != "running":
                return state
            time.sleep(0.02)
        self.fail("run did not finish")


class BackendConfig(unittest.TestCase):
    def env(self, **values):
        with mock.patch.dict(os.environ, values, clear=True):
            return server.Config.from_env()

    def test_default_is_lm_studio(self):
        config = self.env()
        self.assertEqual((config.backend, config.backend_url, config.backend_name),
                         ("lmstudio", "http://localhost:1234", "LM Studio"))

    def test_ollama_and_vllm_have_their_own_variables_and_default_ports(self):
        self.assertEqual(self.env(BACKEND="ollama").backend_url, "http://localhost:11434")
        self.assertEqual(self.env(BACKEND="vllm").backend_url, "http://localhost:8000")
        config = self.env(BACKEND="ollama", OLLAMA_HOST="192.0.2.7", OLLAMA_PORT="11500", OLLAMA_API_KEY="ok",
                          LMSTUDIO_HOST="192.0.2.5", LMSTUDIO_API_KEY="lk")
        self.assertEqual((config.backend_url, config.api_key, config.backend_name), ("http://192.0.2.7:11500", "ok", "Ollama"))
        config = self.env(BACKEND="vllm", VLLM_URL="https://gpu.example/", VLLM_API_KEY="vk")
        self.assertEqual((config.backend_url, config.api_key, config.backend_name), ("https://gpu.example/", "vk", "vLLM"))

    def test_llamacpp_has_its_own_variables(self):
        config = self.env(BACKEND="llamacpp", LLAMACPP_HOST="192.0.2.9", LLAMACPP_API_KEY="lk")
        self.assertEqual((config.backend_url, config.api_key, config.backend_name), ("http://192.0.2.9:8080", "lk", "llama.cpp"))

    def test_backend_name_is_case_and_space_tolerant(self):
        self.assertEqual(self.env(BACKEND="  Ollama ").backend, "ollama")
        self.assertEqual(self.env(BACKEND="").backend, "lmstudio")

    def test_unknown_backend_is_rejected_with_the_choices(self):
        with self.assertRaises(ValueError) as caught:
            self.env(BACKEND="koboldcpp")
        self.assertIn("lmstudio, ollama, vllm, llamacpp", str(caught.exception))

    def test_host_with_port_is_split(self):
        # the form Ollama itself uses for OLLAMA_HOST, e.g. from a shell profile
        self.assertEqual(self.env(BACKEND="ollama", OLLAMA_HOST="127.0.0.1:11434").backend_url, "http://127.0.0.1:11434")
        self.assertEqual(self.env(LMSTUDIO_HOST="192.0.2.5:4321", LMSTUDIO_PORT="1").backend_url, "http://192.0.2.5:4321")
        self.assertEqual(self.env(LMSTUDIO_HOST="fe80::1").backend_url, "http://[fe80::1]:1234")   # IPv6 stays IPv6

    def test_ollama_host_may_be_a_url_like_ollama_accepts(self):
        self.assertEqual(self.env(BACKEND="ollama", OLLAMA_HOST="http://gpu:11434").backend_url, "http://gpu:11434")
        with self.assertRaises(ValueError):
            self.env(BACKEND="vllm", VLLM_HOST="http://gpu:8000")   # vLLM has VLLM_URL for that


class OllamaModels(AppMixin, unittest.TestCase):
    MODELS = {
        "granite3.2-vision:2b": {"caps": ["completion", "vision"], "size": 2 * GB, "text": "Rechnung 1"},
        "moondream:1.8b": {"caps": ["completion", "vision"], "size": 1 * GB, "text": "Rechnung l"},
        "smollm2:135m": {"caps": ["completion"], "size": GB // 10},
        "nomic-embed-text:latest": {"caps": ["embedding"], "size": GB // 4},
        "llava:7b": {"families": ["llama", "clip"], "size": 4 * GB},      # older Ollama: no capabilities field
    }

    def setUp(self):
        self.ollama = FakeOllama({k: dict(v) for k, v in self.MODELS.items()}, loaded=["smollm2:135m"])
        self.addCleanup(self.ollama.close)
        self.base = self.start_app(backend="ollama", backend_url=self.ollama.url)

    def test_only_vision_models_are_offered_with_size_and_quantization(self):
        status, body = self.request(self.base, "/api/models")
        self.assertEqual(status, 200)
        self.assertEqual([m["id"] for m in body["models"]], ["granite3.2-vision:2b", "moondream:1.8b", "llava:7b"])
        first = body["models"][0]
        self.assertEqual((first["size_bytes"], first["params"], first["quantization"], first["loaded"]),
                         (2 * GB, "2B", "Q4_K_M", False))
        self.assertEqual(body["default"], "moondream:1.8b")     # nothing loaded: the smallest vision model
        self.assertEqual(body["backend"], {"id": "ollama", "name": "Ollama", "can_load": True, "loading_allowed": True})

    def test_loaded_models_come_from_api_ps_with_their_memory_size(self):
        _, body = self.request(self.base, "/api/models")
        self.assertEqual(body["loaded_now"], [{"id": "smollm2:135m", "kind": "text", "size_bytes": 2 * (GB // 10),
                                               "params": "2B", "quantization": "Q4_K_M"}])

    def test_show_is_asked_once_per_model_version(self):
        self.request(self.base, "/api/models")
        self.request(self.base, "/api/models")
        shows = [p for m, p in self.ollama.paths("POST") if p == "/api/show"]
        self.assertEqual(len(shows), len(self.MODELS))

    def test_a_model_whose_show_fails_is_left_out(self):
        self.ollama.models["broken:1b"] = {"caps": ["vision"]}
        original = self.ollama.route

        def route(method, path, body):
            if path == "/api/show" and body.get("model") == "broken:1b":
                return 500, {"error": "boom"}
            return original(method, path, body)
        self.ollama.route = route
        status, body = self.request(self.base, "/api/models")
        self.assertEqual(status, 200)
        self.assertNotIn("broken:1b", [m["id"] for m in body["models"]])

    def test_analyze_uses_native_chat_with_raw_base64_images(self):
        status, body = self.request(self.base, "/api/analyze?mode=ocr&model=granite3.2-vision:2b", PNG)
        self.assertEqual((status, body["text"], body["model"]), (200, "Rechnung 1", "granite3.2-vision:2b"))
        chat = [b for m, p, b, _ in self.ollama.requests if p == "/api/chat"][0]
        self.assertEqual(chat["stream"], False)
        self.assertEqual(chat["options"]["temperature"], 0)
        message = chat["messages"][0]
        self.assertEqual(message["images"], [base64.b64encode(PNG).decode()])
        self.assertIn("Extract all text", message["content"])

    def test_chat_error_is_reported_with_the_backend_name(self):
        self.ollama.models["moondream:1.8b"]["chat_error"] = "model requires more system memory"
        status, body = self.request(self.base, "/api/analyze?mode=ocr&model=moondream:1.8b", PNG)
        self.assertEqual(status, 502)
        self.assertIn("Ollama", body["error"])
        self.assertIn("more system memory", body["error"])

    def test_unreachable_ollama_names_ollama(self):
        base = self.start_app(backend="ollama", backend_url="http://127.0.0.1:9")
        status, body = self.request(base, "/api/models")
        self.assertEqual(status, 502)
        self.assertIn("Ollama is not reachable", body["error"])


class OllamaBenchmark(AppMixin, unittest.TestCase):
    def setUp(self):
        self._saved = (benchmark.UNLOAD_VERIFY_ATTEMPTS, benchmark.UNLOAD_VERIFY_DELAY)
        benchmark.UNLOAD_VERIFY_ATTEMPTS, benchmark.UNLOAD_VERIFY_DELAY = 3, 0.01
        self.addCleanup(setattr, benchmark, "UNLOAD_VERIFY_ATTEMPTS", self._saved[0])
        self.addCleanup(setattr, benchmark, "UNLOAD_VERIFY_DELAY", self._saved[1])
        self.ollama = FakeOllama({
            "a:2b": {"caps": ["vision"], "text": "Rechnung Nr. 1", "load_ns": 1_500_000_000},
            "b:1b": {"caps": ["vision"], "text": "Rechnung Nr. 1"},
            "t:1b": {"caps": ["completion"]},
        }, loaded=["t:1b", "b:1b"])
        self.addCleanup(self.ollama.close)
        self.base = self.start_app(backend="ollama", backend_url=self.ollama.url)

    def test_every_model_is_loaded_alone_and_everything_is_unloaded_at_the_end(self):
        state = self.run_benchmark(self.base, ["a:2b", "b:1b"])
        self.assertEqual(state["status"], "done", state)
        loads = [e for e in self.ollama.events if e[0] == "load"]
        self.assertEqual(loads, [("load", "a:2b", []), ("load", "b:1b", [])])
        self.assertEqual(self.ollama.loaded, [])
        self.assertEqual([r["status"] for r in state["results"]], ["ok", "ok"])
        self.assertEqual(state["results"][0]["reported_load_seconds"], 1.5)    # load_duration is in nanoseconds
        self.assertIsNone(state["results"][1]["reported_load_seconds"])
        self.assertIsNotNone(state["results"][1]["load_seconds"])

    def test_restore_loads_the_previous_models_again(self):
        state = self.run_benchmark(self.base, ["a:2b"], restore=True)
        self.assertEqual(sorted(state["restore"]["loaded"]), ["b:1b", "t:1b"])
        self.assertEqual(sorted(self.ollama.loaded), ["b:1b", "t:1b"])


class OllamaModelNames(AppMixin, unittest.TestCase):
    """Ollama names are name:tag. A name without its tag, or a text model, must not reach /api/chat as is."""

    def setUp(self):
        self.ollama = FakeOllama({
            "qwen3.8-27b-uncensored:q4_k_m": {"caps": ["completion"], "size": 18 * GB},
            "granite3.2-vision:2b": {"caps": ["completion", "vision"], "size": 2 * GB, "text": "Rechnung 1"},
            "llava:7b": {"caps": ["vision"], "size": 4 * GB, "text": "Rechnung 7"},
            "llava:13b": {"caps": ["vision"], "size": 8 * GB, "text": "Rechnung 13"},
        }, loaded=["qwen3.8-27b-uncensored:q4_k_m"])
        self.addCleanup(self.ollama.close)

    def chats(self):
        return [b["model"] for m, p, b, _ in self.ollama.requests if p == "/api/chat"]

    def test_a_default_that_is_a_text_model_is_not_used_and_the_page_is_told(self):
        base = self.start_app(backend="ollama", backend_url=self.ollama.url, default_model="qwen3.8-27b-uncensored")
        status, body = self.request(base, "/api/models")
        self.assertEqual(status, 200)
        self.assertEqual(body["default"], "granite3.2-vision:2b")    # the smallest vision model instead
        self.assertIn("DEFAULT_MODEL", body["notice"])
        self.assertIn("not a vision model", body["notice"])
        status, answer = self.request(base, "/api/analyze?mode=ocr", PNG)
        self.assertEqual(status, 200, answer)
        self.assertEqual(self.chats(), ["granite3.2-vision:2b"])

    def test_a_default_without_its_tag_is_completed_when_only_one_tag_exists(self):
        base = self.start_app(backend="ollama", backend_url=self.ollama.url, default_model="granite3.2-vision")
        status, body = self.request(base, "/api/models")
        self.assertEqual(body["default"], "granite3.2-vision:2b")
        self.assertNotIn("notice", body)

    def test_an_ambiguous_default_is_not_guessed(self):
        base = self.start_app(backend="ollama", backend_url=self.ollama.url, default_model="llava")
        _, body = self.request(base, "/api/models")
        self.assertEqual(body["default"], "granite3.2-vision:2b")
        self.assertIn("llava:7b", body["notice"])
        self.assertIn("llava:13b", body["notice"])

    def test_a_text_model_sent_directly_is_refused_with_a_clear_message(self):
        base = self.start_app(backend="ollama", backend_url=self.ollama.url)
        for name in ("qwen3.8-27b-uncensored", "qwen3.8-27b-uncensored:q4_k_m", "nope:1b"):
            status, body = self.request(base, "/api/analyze?mode=ocr&model=" + name, PNG)
            self.assertEqual(status, 400, (name, body))
            self.assertIn("not a vision model on Ollama", body["error"])
        self.assertEqual(self.chats(), [])

    def test_a_name_without_its_tag_is_completed_for_the_request(self):
        base = self.start_app(backend="ollama", backend_url=self.ollama.url)
        status, body = self.request(base, "/api/analyze?mode=ocr&model=granite3.2-vision", PNG)
        self.assertEqual(status, 200, body)
        self.assertEqual(body["model"], "granite3.2-vision:2b")
        self.assertEqual(self.chats(), ["granite3.2-vision:2b"])


class ModelLoadingDisabled(AppMixin, unittest.TestCase):
    """The hard-coded lock: every model is listed, only loaded ones can be used, nothing is loaded or unloaded."""

    def setUp(self):
        self.ollama = FakeOllama({
            "big:7b": {"caps": ["vision"], "size": 4 * GB, "text": "Rechnung big"},
            "small:1b": {"caps": ["vision"], "size": GB // 2, "text": "Rechnung small"},
            "other:2b": {"caps": ["vision"], "size": GB, "text": "Rechnung other"},
        }, loaded=["big:7b", "other:2b"])
        self.addCleanup(self.ollama.close)
        self.base = self.start_app(backend="ollama", backend_url=self.ollama.url, allow_loading=False)

    def model_requests(self):
        return [(m, p, (b or {}).get("model")) for m, p, b, _ in self.ollama.requests if p in ("/api/generate", "/api/chat")]

    def test_model_loading_allowed_comes_from_the_environment(self):
        self.assertIs(server.Config().allow_loading, True)
        cases = {None: True, "": True, "true": True, "1": True, " Yes ": True, "on": True,
                 "false": False, "0": False, " No ": False, "OFF": False}
        for value, expected in cases.items():
            env = {} if value is None else {"MODEL_LOADING_ALLOWED": value}
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertIs(server.Config.from_env().allow_loading, expected, repr(value))

    def test_only_the_documented_name_switches_loading(self):
        others = {name: "false" for name in ("ALLOW_MODEL_LOADING", "ALLOW_LOADING", "OLLAMA_ALLOW_LOADING")}
        with mock.patch.dict(os.environ, others, clear=True):
            self.assertIs(server.Config.from_env().allow_loading, True)

    def test_an_unclear_value_stops_the_start_with_its_name(self):
        for value in ("sometimes", "2", "disabled"):
            with mock.patch.dict(os.environ, {"MODEL_LOADING_ALLOWED": value}, clear=True):
                with self.assertRaises(ValueError) as caught:
                    server.Config.from_env()
            self.assertIn("MODEL_LOADING_ALLOWED", str(caught.exception))

    def test_switched_off_in_the_environment_the_running_app_refuses_unloaded_models(self):
        env = {"BACKEND": "ollama", "OLLAMA_URL": self.ollama.url, "MODEL_LOADING_ALLOWED": "false"}
        with mock.patch.dict(os.environ, env, clear=True):
            config = server.Config.from_env()
        app = server.make_server(config, "127.0.0.1", 0)
        threading.Thread(target=app.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        self.addCleanup(app.server_close)
        self.addCleanup(app.shutdown)
        base = "http://127.0.0.1:%d" % app.server_address[1]
        self.assertIs(self.request(base, "/api/models")[1]["backend"]["loading_allowed"], False)
        status, body = self.request(base, "/api/analyze?mode=ocr&model=small:1b", PNG)
        self.assertEqual(status, 403, body)
        self.assertEqual(self.model_requests(), [])
        self.assertEqual(self.request(base, "/api/analyze?mode=ocr&model=other:2b", PNG)[0], 200)

    def test_every_vision_model_is_listed_and_the_default_is_a_loaded_one(self):
        status, body = self.request(self.base, "/api/models")
        self.assertEqual(status, 200)
        self.assertEqual(sorted(m["id"] for m in body["models"]), ["big:7b", "other:2b", "small:1b"])
        self.assertEqual(body["backend"], {"id": "ollama", "name": "Ollama", "can_load": False, "loading_allowed": False})
        self.assertIn(body["default"], ("big:7b", "other:2b"))   # never the smaller, unloaded small:1b

    def test_a_configured_default_that_is_not_loaded_falls_back_to_a_loaded_one(self):
        base = self.start_app(backend="ollama", backend_url=self.ollama.url, allow_loading=False, default_model="small:1b")
        self.assertIn(self.request(base, "/api/models")[1]["default"], ("big:7b", "other:2b"))

    def test_a_loaded_model_answers(self):
        status, body = self.request(self.base, "/api/analyze?mode=ocr&model=other:2b", PNG)
        self.assertEqual(status, 200, body)
        self.assertEqual(body["text"], "Rechnung other")

    def test_a_model_that_is_not_loaded_is_refused_before_the_server_sees_it(self):
        status, body = self.request(self.base, "/api/analyze?mode=ocr&model=small:1b", PNG)
        self.assertEqual(status, 403)
        self.assertIn("small:1b is not loaded", body["error"])
        self.assertIn("disabled", body["error"])
        self.assertEqual(self.model_requests(), [])
        self.assertNotIn("small:1b", self.ollama.loaded)

    def test_an_unknown_model_is_refused_too(self):
        status, body = self.request(self.base, "/api/analyze?mode=ocr&model=nope:1b", PNG)
        self.assertEqual(status, 400)
        self.assertIn("not a vision model", body["error"])
        self.assertEqual(self.model_requests(), [])

    def test_without_any_loaded_model_nothing_is_picked(self):
        self.ollama.loaded.clear()
        status, body = self.request(self.base, "/api/analyze?mode=ocr", PNG)
        self.assertEqual(status, 503)
        self.assertIn("No vision model is loaded", body["error"])
        self.assertEqual(self.model_requests(), [])
        self.assertEqual(self.request(self.base, "/api/models")[1]["default"], "")

    def test_an_unknown_load_state_counts_as_not_loaded(self):
        listing = {"models": [{"id": "m", "loaded": None}], "loaded_now": []}
        with mock.patch.object(server, "fetch_models", return_value=listing), \
                mock.patch.object(backends, "chat") as chat:
            with self.assertRaises(server.ApiError) as caught:
                server.analyze(server.Config(allow_loading=False), PNG, "ocr", "m")
        self.assertEqual(caught.exception.status, 403)
        chat.assert_not_called()

    def test_benchmark_refuses_models_that_are_not_loaded(self):
        query = urllib.parse.urlencode([("mode", "ocr"), ("model", "big:7b"), ("model", "small:1b")])
        status, body = self.request(self.base, "/api/debug/runs?" + query, PNG)
        self.assertEqual(status, 403)
        self.assertIn("small:1b", body["error"])
        self.assertEqual(self.model_requests(), [])

    def test_benchmark_runs_loaded_models_without_unloading_or_loading_anything(self):
        state = self.run_benchmark(self.base, ["big:7b", "other:2b"], restore=True)
        self.assertEqual(state["status"], "done", state)
        self.assertEqual([r["status"] for r in state["results"]], ["ok", "ok"])
        self.assertEqual([r["load_seconds"] for r in state["results"]], [None, None])
        self.assertEqual(self.model_requests(), [("POST", "/api/chat", "big:7b"), ("POST", "/api/chat", "other:2b")])
        self.assertEqual(sorted(self.ollama.loaded), ["big:7b", "other:2b"])
        self.assertEqual(state["restore"]["models"], [])
        self.assertTrue(any("loading is disabled" in line["message"] for line in state["log"]), state["log"])

    def test_benchmark_all_means_all_loaded_models(self):
        status, started = self.request(self.base, "/api/debug/runs?mode=ocr&all=1", PNG)
        self.assertEqual(status, 200, started)
        self.assertEqual(sorted(started["models"]), ["big:7b", "other:2b"])


class TlsFakeOllama(FakeOllama):
    tls = True


class SelfSignedTls(AppMixin, unittest.TestCase):
    """A model server behind HTTPS with a self-signed certificate (e.g. a reverse proxy)."""

    def setUp(self):
        self.ollama = TlsFakeOllama({"granite3.2-vision:2b": {"caps": ["completion", "vision"], "size": GB, "text": "Rechnung 1"}})
        self.addCleanup(self.ollama.close)

    def test_the_certificate_is_checked_by_default(self):
        base = self.start_app(backend="ollama", backend_url=self.ollama.url)
        status, body = self.request(base, "/api/models")
        self.assertEqual(status, 502)
        self.assertIn("CERTIFICATE_VERIFY_FAILED", body["error"])
        self.assertEqual(self.ollama.requests, [], "no request may get through an unverified connection")

    def test_tls_verify_off_lists_the_models(self):
        base = self.start_app(backend="ollama", backend_url=self.ollama.url + "/", tls_verify=False)
        status, body = self.request(base, "/api/models")
        self.assertEqual(status, 200, body)
        self.assertEqual([m["id"] for m in body["models"]], ["granite3.2-vision:2b"])

    def test_tls_verify_off_runs_the_ocr(self):
        base = self.start_app(backend="ollama", backend_url=self.ollama.url, tls_verify=False)
        status, body = self.request(base, "/api/analyze?mode=ocr&model=granite3.2-vision:2b", PNG)
        self.assertEqual(status, 200, body)
        self.assertEqual(body["text"], "Rechnung 1")

    def test_tls_verify_off_still_works_over_plain_http(self):
        plain = FakeOllama({"granite3.2-vision:2b": {"caps": ["completion", "vision"], "size": GB}})
        self.addCleanup(plain.close)
        base = self.start_app(backend="ollama", backend_url=plain.url, tls_verify=False)
        self.assertEqual(self.request(base, "/api/models")[0], 200)


class Vllm(AppMixin, unittest.TestCase):
    def setUp(self):
        self.vllm = FakeVllm({"Qwen/Qwen2.5-VL-7B-Instruct": "Rechnung Nr. 1"}, key="secret")
        self.addCleanup(self.vllm.close)
        self.base = self.start_app(backend="vllm", backend_url=self.vllm.url, api_key="secret")

    def test_models_are_listed_as_served_and_loaded(self):
        status, body = self.request(self.base, "/api/models")
        self.assertEqual(status, 200)
        self.assertEqual(body["models"], [{"id": "Qwen/Qwen2.5-VL-7B-Instruct", "loaded": True, "size_bytes": None,
                                           "params": None, "quantization": None}])
        self.assertEqual(body["loaded_now"], [{"id": "Qwen/Qwen2.5-VL-7B-Instruct", "kind": "unknown",
                                               "size_bytes": None, "params": None, "quantization": None}])
        self.assertEqual(body["backend"], {"id": "vllm", "name": "vLLM", "can_load": False, "loading_allowed": True})

    def test_analyze_sends_a_data_url_and_the_api_key(self):
        status, body = self.request(self.base, "/api/analyze?mode=ocr&model=Qwen/Qwen2.5-VL-7B-Instruct", PNG)
        self.assertEqual((status, body["text"]), (200, "Rechnung Nr. 1"))
        method, path, chat, auth = [r for r in self.vllm.requests if r[1] == "/v1/chat/completions"][0]
        self.assertEqual(auth, "Bearer secret")
        image = chat["messages"][0]["content"][1]["image_url"]["url"]
        self.assertTrue(image.startswith("data:image/png;base64,"))

    def test_wrong_key_is_a_clear_error(self):
        base = self.start_app(backend="vllm", backend_url=self.vllm.url, api_key="wrong")
        status, body = self.request(base, "/api/models")
        self.assertEqual(status, 502)
        self.assertIn("vLLM answered HTTP 401", body["error"])

    def test_an_unlisted_model_is_refused_before_vllm_sees_it(self):
        status, body = self.request(self.base, "/api/analyze?mode=ocr&model=nope", PNG)
        self.assertEqual(status, 400)
        self.assertIn("nope is not a vision model on vLLM", body["error"])
        self.assertNotIn(("POST", "/v1/chat/completions"), self.vllm.paths())

    def test_benchmark_runs_without_loading_or_unloading_and_restores_nothing(self):
        state = self.run_benchmark(self.base, ["Qwen/Qwen2.5-VL-7B-Instruct"], restore=True)
        self.assertEqual(state["status"], "done", state)
        self.assertEqual({p for _, p in self.vllm.paths()}, {"/v1/models", "/v1/chat/completions"})
        result = state["results"][0]
        self.assertEqual((result["status"], result["load_seconds"]), ("ok", None))
        self.assertEqual(result["total_seconds"], result["infer_seconds"])
        self.assertEqual(state["restore"], {"requested": False, "models": [], "loaded": [], "failed": []})


class LlamaCppSingleModel(AppMixin, unittest.TestCase):
    def setUp(self):
        meta = {"n_params": 162974016, "size": 173263104, "ftype": "Q8_0", "n_ctx": 8192}   # from a real b11610
        self.llama = FakeLlamaCpp({"gemma-3-4b-it-Q4_K_M.gguf": {"modalities": ["text", "image", "video"],
                                                                 "text": "Rechnung", "meta": meta}}, router=False)
        self.addCleanup(self.llama.close)
        self.base = self.start_app(backend="llamacpp", backend_url=self.llama.url)

    def test_the_one_model_is_listed_loaded_and_cannot_be_unloaded(self):
        _, body = self.request(self.base, "/api/models")
        self.assertEqual(body["models"], [{"id": "gemma-3-4b-it-Q4_K_M.gguf", "loaded": True, "size_bytes": 173263104,
                                           "params": "163M", "quantization": "Q8_0"}])
        self.assertEqual(body["loaded_now"][0]["kind"], "vision")
        self.assertEqual(body["backend"], {"id": "llamacpp", "name": "llama.cpp", "can_load": False, "loading_allowed": True})

    def test_meta_may_be_null_while_loading(self):
        self.llama.models["gemma-3-4b-it-Q4_K_M.gguf"]["meta"] = None
        _, body = self.request(self.base, "/api/models")
        self.assertEqual((body["models"][0]["size_bytes"], body["models"][0]["params"]), (None, None))

    def test_a_text_only_server_offers_no_vision_model(self):
        self.llama.models["gemma-3-4b-it-Q4_K_M.gguf"]["modalities"] = ["text"]
        _, body = self.request(self.base, "/api/models")
        self.assertEqual((body["models"], body["loaded_now"][0]["kind"]), ([], "text"))

    def test_analyze_and_benchmark_without_load(self):
        status, body = self.request(self.base, "/api/analyze?mode=ocr&model=gemma-3-4b-it-Q4_K_M.gguf", PNG)
        self.assertEqual((status, body["text"]), (200, "Rechnung"))
        state = self.run_benchmark(self.base, ["gemma-3-4b-it-Q4_K_M.gguf"], restore=True)
        self.assertEqual((state["status"], state["results"][0]["status"], state["results"][0]["load_seconds"]),
                         ("done", "ok", None))
        self.assertEqual([e[0] for e in self.llama.events], ["chat", "chat"])


class LlamaCppRouter(AppMixin, unittest.TestCase):
    MODELS = {
        "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M": {"modalities": ["text", "image"], "text": "Rechnung Nr. 1"},
        "ggml-org/SmolVLM-256M-Instruct-GGUF:Q8_0": {"modalities": ["text", "image"], "text": "Rechnung Nr. 1"},
        "ggml-org/Qwen3-0.6B-GGUF:Q8_0": {"modalities": ["text"], "text": "x"},
        "unknown/model:Q4": {"text": "Rechnung Nr. 7"},       # not loaded yet: modalities unknown
    }

    def setUp(self):
        self._saved = (benchmark.UNLOAD_VERIFY_ATTEMPTS, benchmark.UNLOAD_VERIFY_DELAY)
        benchmark.UNLOAD_VERIFY_ATTEMPTS, benchmark.UNLOAD_VERIFY_DELAY = 40, 0.01
        self.addCleanup(setattr, benchmark, "UNLOAD_VERIFY_ATTEMPTS", self._saved[0])
        self.addCleanup(setattr, benchmark, "UNLOAD_VERIFY_DELAY", self._saved[1])
        self.llama = FakeLlamaCpp({k: dict(v) for k, v in self.MODELS.items()}, router=True,
                                  loaded=["ggml-org/Qwen3-0.6B-GGUF:Q8_0"])
        self.addCleanup(self.llama.close)
        self.base = self.start_app(backend="llamacpp", backend_url=self.llama.url)

    def test_router_lists_vision_and_unknown_models_with_their_status(self):
        _, body = self.request(self.base, "/api/models")
        self.assertEqual([(m["id"], m["loaded"]) for m in body["models"]], [
            ("ggml-org/gemma-3-4b-it-GGUF:Q4_K_M", False), ("ggml-org/SmolVLM-256M-Instruct-GGUF:Q8_0", False),
            ("unknown/model:Q4", False)])
        self.assertEqual([(m["id"], m["kind"]) for m in body["loaded_now"]], [("ggml-org/Qwen3-0.6B-GGUF:Q8_0", "text")])
        self.assertTrue(body["backend"]["can_load"])

    def test_benchmark_waits_for_each_load_and_unloads_everything_before(self):
        models = ["ggml-org/gemma-3-4b-it-GGUF:Q4_K_M", "ggml-org/SmolVLM-256M-Instruct-GGUF:Q8_0"]
        state = self.run_benchmark(self.base, models, restore=True)
        self.assertEqual(state["status"], "done", state["log"])
        loads = [e for e in self.llama.events if e[0] == "load"]
        self.assertEqual([(e[1], e[2]) for e in loads[:2]], [(models[0], []), (models[1], [])])
        for result in state["results"]:
            self.assertEqual(result["status"], "ok", result)
            self.assertGreaterEqual(result["load_seconds"], 0.05)       # waited until the status said loaded
        self.assertEqual(state["restore"]["loaded"], ["ggml-org/Qwen3-0.6B-GGUF:Q8_0"])
        deadline = time.monotonic() + 2
        while self.llama.status["ggml-org/Qwen3-0.6B-GGUF:Q8_0"] != "loaded" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual([m for m, v in self.llama.status.items() if v == "loaded"], ["ggml-org/Qwen3-0.6B-GGUF:Q8_0"])

    def test_a_model_that_fails_to_load_is_recorded_and_the_run_goes_on(self):
        self.llama.models["ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"]["fail"] = True
        state = self.run_benchmark(self.base, ["ggml-org/gemma-3-4b-it-GGUF:Q4_K_M", "unknown/model:Q4"])
        first, second = state["results"]
        self.assertEqual(first["status"], "error")
        self.assertIn("failed to load", first["error"])
        self.assertEqual((second["status"], second["text"]), ("ok", "Rechnung Nr. 7"))


if __name__ == "__main__":
    unittest.main()
