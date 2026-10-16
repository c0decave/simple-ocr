"""Model servers behind one small interface: LM Studio, Ollama and vLLM.

Every function takes the Config and talks to config.backend_url. What differs per server:

- listing: LM Studio /api/v1/models (older: /api/v0/models, /v1/models); Ollama /api/tags plus
  /api/show (capabilities) and /api/ps (loaded); vLLM /v1/models (everything listed is served);
  llama.cpp GET /models (router mode, with a status per model) or /v1/models (one fixed model).
- chat: LM Studio, vLLM and llama.cpp /v1/chat/completions with a data: URL; Ollama /api/chat with "images".
- load and unload (benchmark only): LM Studio /api/v1/models/load|unload; Ollama /api/generate
  with an empty prompt, keep_alive 0 unloads; llama.cpp router POST /models/load|unload (they finish
  in the background, so the status is polled); vLLM and a single-model llama.cpp cannot do either.
"""
import base64
import http.client
import json
import os
import socket
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request

# id -> (display name, prefix of its .env variables, default port, can load and unload models;
#        None = only in llama.cpp's router mode, which is asked at run time)
BACKENDS = {
    "lmstudio": ("LM Studio", "LMSTUDIO", 1234, True),
    "ollama": ("Ollama", "OLLAMA", 11434, True),
    "vllm": ("vLLM", "VLLM", 8000, False),
    "llamacpp": ("llama.cpp", "LLAMACPP", 8080, None),
}
LOAD_POLL_DELAY = 0.1   # seconds between status checks while llama.cpp loads a model

# Hard-coded on purpose, not an .env setting. False: the web interface never makes the model server
# load (or unload) models; every model is listed, but only the loaded ones can be used, also in the
# benchmark. True: models are loaded on first use, and the benchmark unloads and loads them.
MODEL_LOADING_ALLOWED = True


def can_load(config):
    fixed = BACKENDS[config.backend][3]
    return fixed if fixed is not None else _llamacpp_listing(config)[1]


def manages_models(config):
    """True when the benchmark unloads and loads models: the server can, and loading is allowed."""
    return config.allow_loading and can_load(config)


def describe(config):
    """What the browser may know about the backend: never the URL or key."""
    return {"id": config.backend, "name": config.backend_name, "can_load": manages_models(config),
            "loading_allowed": config.allow_loading}


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


class NotSupported(ApiError):
    """The server answered, but not usefully for this endpoint, so a different endpoint may still work."""


def error_message(body):
    """The message of an error body, else None.

    LM Studio and Ollama: {"error": "text"} or {"error": {"message": ...}}; vLLM: {"object": "error", "message": ...}.
    """
    if not isinstance(body, dict):
        return None
    if "error" in body:
        error = body["error"]
        if isinstance(error, dict):
            return str(error.get("message") or error.get("type") or json.dumps(error))
        return str(error)
    if body.get("object") == "error" and body.get("message"):
        return str(body["message"])
    return None


def _unverified_tls():
    """For <PREFIX>_TLS_VERIFY=false: still TLS, but any certificate (self-signed) is accepted."""
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Redirects are not followed: urllib would send the Authorization header to the new host too."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open(config, request):
    context = None if config.tls_verify else _unverified_tls()
    opener = urllib.request.build_opener(_NoRedirect, urllib.request.HTTPSHandler(context=context))
    return opener.open(request, timeout=config.timeout)


def _reason_for_browser(reason, prefix):
    """Why the server is not reachable, without host names or addresses (those go to the log only)."""
    if isinstance(reason, ssl.SSLCertVerificationError):
        return ("the TLS certificate is not trusted (CERTIFICATE_VERIFY_FAILED); for a self-signed certificate "
                "set %s_TLS_VERIFY=false" % prefix)
    if isinstance(reason, ssl.SSLError):
        return "TLS error (%s)" % (reason.reason or type(reason).__name__)
    if isinstance(reason, socket.gaierror):
        return "the host name was not found"
    if isinstance(reason, OSError) and reason.errno:
        return os.strerror(reason.errno)
    return type(reason).__name__


def upstream_call(config, path, payload=None):
    """Call the model server and return the decoded JSON. Every failure becomes an ApiError(502/504)."""
    name = config.backend_name
    url = config.backend_url.rstrip("/") + path
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = "Bearer " + config.api_key
    prefix = BACKENDS[config.backend][1]
    try:
        request = urllib.request.Request(
            url, data=data, headers=headers, method="GET" if payload is None else "POST",
        )
    except ValueError:
        raise ApiError(502, "The %s address is not a valid URL; check %s_URL" % (name, prefix))
    try:
        with _open(config, request) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        if 300 <= error.code < 400:
            raise ApiError(502, "%s answered with a redirect (HTTP %d), which is not followed; set %s_URL to the "
                                "address it points to" % (name, error.code, prefix))
        detail = error.read(2000).decode("utf-8", "replace")
        try:
            detail = error_message(json.loads(detail)) or detail
        except ValueError:
            pass
        raise NotSupported(502, "%s answered HTTP %d: %s" % (name, error.code, detail[:500]))
    except (socket.timeout, TimeoutError):
        raise ApiError(504, "%s did not answer within %.0f s" % (name, config.timeout))
    except urllib.error.URLError as error:
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            raise ApiError(504, "%s did not answer within %.0f s" % (name, config.timeout))
        # the URL goes to the log only: the browser never learns where the model server is
        print("simple-ocr: %s at %s: %s" % (name, config.backend_url, error.reason), file=sys.stderr, flush=True)
        raise ApiError(502, "%s is not reachable: %s" % (name, _reason_for_browser(error.reason, prefix)))
    except ValueError:
        raise NotSupported(502, "%s returned invalid JSON for %s" % (name, path))
    except (OSError, http.client.HTTPException) as error:
        # e.g. connection reset or truncated body while reading the reply
        raise ApiError(502, "Connection to %s failed while reading %s: %s" % (name, path, error))
    if not isinstance(result, dict):
        raise NotSupported(502, "%s returned an unexpected JSON shape for %s" % (name, path))
    return result


def post(config, path, payload):
    """POST that also treats an "error" in a 200 body as a failure (LM Studio does that)."""
    reply = upstream_call(config, path, payload)
    message = error_message(reply)
    if message:
        raise ApiError(502, message)
    return reply


# ---- listing helpers

def _entries(config, listing, field_name, id_field):
    """The dict entries of a listing that carry a string id; everything else is ignored."""
    data = listing.get(field_name, [])
    if not isinstance(data, list):
        raise NotSupported(502, "%s returned an unexpected model listing" % config.backend_name)
    return [m for m in data if isinstance(m, dict) and isinstance(m.get(id_field), str)]


def _is_embedding(entry, model_id):
    """Embedding models can have a vision capability but cannot chat, so they must not be offered."""
    return entry.get("type") == "embedding" or "embed" in model_id.lower()


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _text(value):
    return value if isinstance(value, str) else None


# ---- LM Studio

def _models_v1(config):
    """LM Studio's native listing: has size, parameter count, quantization and loaded instances."""
    models, loaded_now = [], []
    for m in _entries(config, upstream_call(config, "/api/v1/models"), "models", "key"):
        loaded = bool(m.get("loaded_instances"))
        capabilities = m.get("capabilities")
        vision = isinstance(capabilities, dict) and capabilities.get("vision") is True
        if _is_embedding(m, m["key"]):
            kind = "embedding"
        else:
            kind = "vision" if vision else "text"
        quantization = m.get("quantization")
        if isinstance(quantization, dict):
            quantization = quantization.get("name")
        entry = {
            "id": m["key"], "size_bytes": _number(m.get("size_bytes")),
            "params": _text(m.get("params_string")), "quantization": _text(quantization),
        }
        if loaded:
            loaded_now.append(dict(entry, kind=kind))
        if kind == "vision":
            models.append(dict(entry, loaded=loaded))
    return models, loaded_now


def _kind_v0(entry):
    if entry.get("type") == "embeddings" or _is_embedding(entry, entry["id"]):
        return "embedding"
    return "vision" if entry.get("type") == "vlm" else "text"


def _models_v0(config):
    """Older LM Studio: vision models and load state, but no size."""
    models, loaded_now = [], []
    for m in _entries(config, upstream_call(config, "/api/v0/models"), "data", "id"):
        loaded = m.get("state") == "loaded"
        if loaded:
            loaded_now.append({"id": m["id"], "size_bytes": None, "params": None,
                               "quantization": None, "kind": _kind_v0(m)})
        if m.get("type") == "vlm" and not _is_embedding(m, m["id"]):
            models.append({"id": m["id"], "loaded": loaded, "size_bytes": None,
                           "params": None, "quantization": None})
    return models, loaded_now


def _models_openai(config):
    """Plain OpenAI-compatible server: all model ids, nothing else is known."""
    models = [{"id": m["id"], "loaded": None, "size_bytes": None, "params": None, "quantization": None}
              for m in _entries(config, upstream_call(config, "/v1/models"), "data", "id")]
    return models, []


def _models_lmstudio(config):
    """Falls back to older endpoints only when LM Studio answered unusably; an unreachable host fails at once."""
    last_error = None
    for loader in (_models_v1, _models_v0, _models_openai):
        try:
            return loader(config)
        except NotSupported as error:
            last_error = error
    raise last_error


# ---- Ollama

_show_cache = {}   # (url, name, digest) -> kind; /api/show is asked once per model version
_show_lock = threading.Lock()


def _ollama_kind(name, info):
    capabilities = info.get("capabilities")
    if isinstance(capabilities, list):
        if "embedding" in capabilities:
            return "embedding"
        return "vision" if "vision" in capabilities else "text"
    # Ollama before capabilities existed: a CLIP projector means vision
    details = info.get("details") if isinstance(info.get("details"), dict) else {}
    families = details.get("families") if isinstance(details.get("families"), list) else []
    if "embed" in name.lower():
        return "embedding"
    return "vision" if {"clip", "mllama"} & set(families) else "text"


def _ollama_show_kind(config, name, digest):
    key = (config.backend_url, name, digest)
    with _show_lock:
        if key in _show_cache:
            return _show_cache[key]
    try:
        kind = _ollama_kind(name, upstream_call(config, "/api/show", {"model": name}))
    except ApiError:
        return None   # e.g. removed between listing and show: leave the model out, do not fail the listing
    with _show_lock:
        _show_cache[key] = kind
    return kind


def _ollama_entry(m):
    details = m.get("details") if isinstance(m.get("details"), dict) else {}
    return {"id": m["name"], "size_bytes": _number(m.get("size")),
            "params": _text(details.get("parameter_size")), "quantization": _text(details.get("quantization_level"))}


def _models_ollama(config):
    tags = _entries(config, upstream_call(config, "/api/tags"), "models", "name")
    running = _entries(config, upstream_call(config, "/api/ps"), "models", "name")
    loaded = {m["name"] for m in running}
    kinds = {m["name"]: _ollama_show_kind(config, m["name"], m.get("digest")) for m in tags}
    models = [dict(_ollama_entry(m), loaded=m["name"] in loaded) for m in tags if kinds[m["name"]] == "vision"]
    # /api/ps reports the memory a loaded model takes, which is what the loaded table is about
    loaded_now = [dict(_ollama_entry(m), kind=kinds.get(m["name"]) or "unknown") for m in running]
    return models, loaded_now


# ---- vLLM

def _models_vllm(config):
    """vLLM serves the models it was started with; nothing says which of them can see images."""
    models, loaded_now = [], []
    for m in _entries(config, upstream_call(config, "/v1/models"), "data", "id"):
        entry = {"id": m["id"], "size_bytes": None, "params": None, "quantization": None}
        models.append(dict(entry, loaded=True))
        loaded_now.append(dict(entry, kind="unknown"))
    return models, loaded_now


# ---- llama.cpp

def _llamacpp_listing(config):
    """(entries, router mode). The router lists every model with a status; a plain server has one model."""
    try:
        entries = _entries(config, upstream_call(config, "/models"), "data", "id")
    except NotSupported:
        entries = _entries(config, upstream_call(config, "/v1/models"), "data", "id")
    return entries, any(isinstance(m.get("status"), dict) for m in entries)


def _llamacpp_status(entry):
    status = entry.get("status")
    return status if isinstance(status, dict) else {}


def _llamacpp_kind(entry):
    architecture = entry.get("architecture")
    modalities = architecture.get("input_modalities") if isinstance(architecture, dict) else None
    if not isinstance(modalities, list):
        return "unknown"   # e.g. a router model that was never loaded
    return "vision" if "image" in modalities else "text"


def _param_count(value):
    """162974016 -> "163M", 7615616512 -> "7.6B", like the parameter strings of the other servers."""
    if value is None or value <= 0:
        return None
    if value >= 999.5e6:   # would round to "1000M"
        return ("%.1f" % (value / 1e9)).rstrip("0").rstrip(".") + "B"
    return "%dM" % round(value / 1e6)


def _models_llamacpp(config):
    entries, router = _llamacpp_listing(config)
    models, loaded_now = [], []
    for m in entries:
        kind = _llamacpp_kind(m)
        loaded = _llamacpp_status(m).get("value") == "loaded" if router else True
        meta = m.get("meta") if isinstance(m.get("meta"), dict) else {}   # null while a model is loading
        entry = {"id": m["id"], "size_bytes": _number(meta.get("size")),
                 "params": _param_count(_number(meta.get("n_params"))), "quantization": _text(meta.get("ftype"))}
        if loaded:
            loaded_now.append(dict(entry, kind=kind))
        if kind in ("vision", "unknown"):
            models.append(dict(entry, loaded=loaded))
    return models, loaded_now


def _llamacpp_wait_loaded(config, model):
    deadline = time.monotonic() + config.timeout
    while True:
        entries, _ = _llamacpp_listing(config)
        status = next((_llamacpp_status(m) for m in entries if m["id"] == model), {})
        if status.get("value") == "loaded":
            return
        if status.get("failed"):
            raise ApiError(502, "%s failed to load %s (exit code %s)" % (config.backend_name, model, status.get("exit_code")))
        if time.monotonic() > deadline:
            raise ApiError(504, "%s did not finish loading %s within %.0f s" % (config.backend_name, model, config.timeout))
        time.sleep(LOAD_POLL_DELAY)


def _llamacpp_post(config, path, model):
    reply = post(config, path, {"model": model})
    if reply.get("success") is False:
        raise ApiError(502, "%s refused %s for %s" % (config.backend_name, path, model))


def list_models(config):
    """(vision models, everything loaded right now with its kind)."""
    loader = {"lmstudio": _models_lmstudio, "ollama": _models_ollama, "vllm": _models_vllm,
              "llamacpp": _models_llamacpp}[config.backend]
    return loader(config)


# ---- chat

def chat(config, model, prompt, mime, image):
    """Send one image with the prompt; returns the reply text (may be None or empty)."""
    encoded = base64.b64encode(image).decode()
    name = config.backend_name
    if config.backend == "ollama":
        reply = upstream_call(config, "/api/chat", {
            "model": model, "stream": False, "options": {"temperature": 0},
            "messages": [{"role": "user", "content": prompt, "images": [encoded]}],
        })
        try:
            content = reply["message"]["content"]
        except (KeyError, TypeError):
            raise ApiError(502, "%s reply has no message content" % name)
    else:
        reply = upstream_call(config, "/v1/chat/completions", {
            "model": model, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, encoded)}},
            ]}],
        })
        try:
            content = reply["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ApiError(502, "%s reply has no message content" % name)
    if content is not None and not isinstance(content, str):
        raise ApiError(502, "%s reply has an unexpected content type" % name)
    return content


# ---- load and unload (only where can_load() is true)

def loaded_ids(config):
    """What has to be unloaded: LM Studio instance ids, Ollama model names."""
    if config.backend == "ollama":
        return [m["name"] for m in _entries(config, upstream_call(config, "/api/ps"), "models", "name")]
    if config.backend == "llamacpp":
        return [m["id"] for m in _llamacpp_listing(config)[0] if _llamacpp_status(m).get("value") in ("loaded", "loading")]
    listing = upstream_call(config, "/api/v1/models")
    models = listing.get("models") if isinstance(listing.get("models"), list) else []
    return [
        instance["id"]
        for model in models if isinstance(model, dict)
        for instance in (model.get("loaded_instances") or [])
        if isinstance(instance, dict) and isinstance(instance.get("id"), str)
    ]


def unload(config, ident):
    if config.backend == "ollama":
        post(config, "/api/generate", {"model": ident, "keep_alive": 0})
    elif config.backend == "llamacpp":
        _llamacpp_post(config, "/models/unload", ident)
    else:
        post(config, "/api/v1/models/unload", {"instance_id": ident})


def load(config, model):
    """Load the model with the server's default settings. Returns the load time the server reports, or None."""
    if config.backend == "ollama":
        reply = post(config, "/api/generate", {"model": model})   # an empty prompt only loads (docs/api.md)
        reported = _number(reply.get("load_duration"))
        return round(reported / 1e9, 3) if reported is not None else None   # Ollama durations are nanoseconds
    if config.backend == "llamacpp":
        _llamacpp_post(config, "/models/load", model)
        _llamacpp_wait_loaded(config, model)
        return None
    reply = post(config, "/api/v1/models/load", {"model": model})
    return _number(reply.get("load_time_seconds"))
