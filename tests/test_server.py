"""Tests for the simple-ocr server, run against a fake LM Studio upstream."""
import base64
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from app import backends, server

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 16

V0_MODELS = {
    "data": [
        {"id": "text-only", "type": "llm", "state": "loaded"},
        {"id": "vision-a", "type": "vlm", "state": "not-loaded"},
        {"id": "vision-b", "type": "vlm", "state": "loaded"},
        {"id": "embed", "type": "embeddings", "state": "not-loaded"},
    ]
}


class FakeUpstream:
    """A scriptable stand-in for LM Studio that records every request."""

    def __init__(self):
        self.requests = []
        self.headers = []
        self.routes = {}  # (method, path) -> (status, body | callable)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _serve(self, method):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                body = json.loads(raw) if raw else None
                outer.requests.append((method, self.path, body))
                outer.headers.append(dict(self.headers))
                status, payload = outer.routes.get((method, self.path), (404, {"error": "no route"}))
                if callable(payload):
                    payload(self)
                    return
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._serve("GET")

            def do_POST(self):
                self._serve("POST")

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def chat_reply(text):
    return (200, {"choices": [{"message": {"role": "assistant", "content": text}}]})


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.upstream = FakeUpstream()
        self.upstream.routes[("GET", "/api/v0/models")] = (200, V0_MODELS)
        self.upstream.routes[("POST", "/v1/chat/completions")] = chat_reply("hello text")
        self.apps = []

    def tearDown(self):
        self.upstream.close()
        for app in self.apps:
            app.shutdown()
            app.server_close()

    def start(self, **overrides):
        # these tests cover the loading mode; the hard-coded default (off) is tested in test_backends
        settings = dict(backend_url=self.upstream.url, timeout=5, allow_loading=True)
        settings.update(overrides)
        app = server.make_server(server.Config(**settings), "127.0.0.1", 0)
        threading.Thread(target=app.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        self.apps.append(app)
        return "http://127.0.0.1:%d" % app.server_address[1]

    def call(self, base, path, body=None):
        request = urllib.request.Request(base + path, data=body, method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def call_json(self, base, path, body=None):
        status, raw = self.call(base, path, body)
        return status, json.loads(raw)

    def chat_requests(self):
        return [r for r in self.upstream.requests if r[1] == "/v1/chat/completions"]


class AnalyzeHappyPath(AppTestCase):
    def test_ocr_is_default_and_sends_image_as_data_url(self):
        base = self.start()
        status, result = self.call_json(base, "/api/analyze", PNG)
        self.assertEqual(status, 200)
        self.assertEqual(result["text"], "hello text")
        self.assertEqual(result["mode"], "ocr")
        (_, _, sent), = self.chat_requests()
        parts = sent["messages"][-1]["content"]
        text_part = next(p for p in parts if p["type"] == "text")
        image_part = next(p for p in parts if p["type"] == "image_url")
        self.assertIn("text", text_part["text"].lower())
        self.assertEqual(
            image_part["image_url"]["url"],
            "data:image/png;base64," + base64.b64encode(PNG).decode(),
        )
        self.assertEqual(sent["temperature"], 0)

    def test_ocr_prompt_has_no_empty_image_clause(self):
        # Measured 2026-10-13 on minicpm-o-2_6: adding "if there is no text, output ..." made the
        # model invent "Title:" / "Company Information:" headings on 2 of 2 runs, none without it.
        self.assertNotIn("no text", server.PROMPTS["ocr"].lower())

    def test_describe_mode_uses_a_different_prompt(self):
        base = self.start()
        self.call_json(base, "/api/analyze?mode=ocr", PNG)
        self.call_json(base, "/api/analyze?mode=describe", PNG)
        ocr, describe = self.chat_requests()
        ocr_prompt = ocr[2]["messages"][-1]["content"][0]["text"]
        describe_prompt = describe[2]["messages"][-1]["content"][0]["text"]
        self.assertNotEqual(ocr_prompt, describe_prompt)
        self.assertIn("describe", describe_prompt.lower())

    def test_default_model_is_first_loaded_vision_model(self):
        base = self.start()
        _, result = self.call_json(base, "/api/analyze", PNG)
        self.assertEqual(result["model"], "vision-b")
        self.assertEqual(self.chat_requests()[0][2]["model"], "vision-b")

    def test_configured_default_model_wins_over_loaded(self):
        base = self.start(default_model="vision-a")
        _, result = self.call_json(base, "/api/analyze", PNG)
        self.assertEqual(result["model"], "vision-a")

    def test_explicit_model_parameter_wins(self):
        base = self.start(default_model="vision-a")
        _, result = self.call_json(base, "/api/analyze?model=vision-b", PNG)
        self.assertEqual(result["model"], "vision-b")

    def test_jpeg_and_webp_are_accepted(self):
        base = self.start()
        for blob, mime in ((JPEG, "image/jpeg"), (WEBP, "image/webp")):
            status, _ = self.call_json(base, "/api/analyze", blob)
            self.assertEqual(status, 200)
            self.assertTrue(self.chat_requests()[-1][2]["messages"][-1]["content"][1]["image_url"]["url"].startswith("data:%s;base64," % mime))


class AnalyzeSadPath(AppTestCase):
    def test_unknown_mode_is_400(self):
        status, result = self.call_json(self.start(), "/api/analyze?mode=poem", PNG)
        self.assertEqual(status, 400)
        self.assertIn("mode", result["error"])
        self.assertEqual(self.chat_requests(), [])

    def test_empty_body_is_400(self):
        status, _ = self.call_json(self.start(), "/api/analyze", b"")
        self.assertEqual(status, 400)

    def test_non_image_body_is_415(self):
        status, result = self.call_json(self.start(), "/api/analyze", b"%PDF-1.7 not an image")
        self.assertEqual(status, 415)
        self.assertIn("PNG", result["error"])
        self.assertEqual(self.chat_requests(), [])

    def test_oversized_body_is_413(self):
        base = self.start(max_upload_bytes=64)
        status, _ = self.call_json(base, "/api/analyze", PNG + b"\x00" * 200)
        self.assertEqual(status, 413)
        self.assertEqual(self.chat_requests(), [])

    def test_upstream_unreachable_is_502(self):
        base = self.start(backend_url="http://127.0.0.1:9", default_model="vision-a")
        status, result = self.call_json(base, "/api/analyze", PNG)
        self.assertEqual(status, 502)
        self.assertIn("LM Studio", result["error"])

    def test_upstream_error_message_is_passed_on(self):
        self.upstream.routes[("POST", "/v1/chat/completions")] = (400, {"error": "model does not support images"})
        status, result = self.call_json(self.start(), "/api/analyze", PNG)
        self.assertEqual(status, 502)
        self.assertIn("model does not support images", result["error"])

    def test_upstream_without_choices_is_502(self):
        self.upstream.routes[("POST", "/v1/chat/completions")] = (200, {"choices": []})
        status, _ = self.call_json(self.start(), "/api/analyze", PNG)
        self.assertEqual(status, 502)

    def test_no_vision_model_available_is_503(self):
        self.upstream.routes[("GET", "/api/v0/models")] = (200, {"data": [{"id": "t", "type": "llm", "state": "loaded"}]})
        status, result = self.call_json(self.start(), "/api/analyze", PNG)
        self.assertEqual(status, 503)
        self.assertIn("vision", result["error"].lower())


class AnalyzeEdgeCases(AppTestCase):
    def test_think_block_is_stripped(self):
        self.upstream.routes[("POST", "/v1/chat/completions")] = chat_reply("<think>hmm\nlet me see</think>\n\nINVOICE 42\n")
        _, result = self.call_json(self.start(), "/api/analyze", PNG)
        self.assertEqual(result["text"], "INVOICE 42")

    def test_doc_wrapper_from_granite_is_stripped(self):
        self.upstream.routes[("POST", "/v1/chat/completions")] = chat_reply("<doc>\n  Invoice 42\n  Total 9,00\n</doc>")
        _, result = self.call_json(self.start(), "/api/analyze", PNG)
        self.assertEqual(result["text"], "Invoice 42\n  Total 9,00")

    def test_doc_tag_inside_text_is_left_alone(self):
        self.upstream.routes[("POST", "/v1/chat/completions")] = chat_reply("use <doc> tags like <doc>this</doc> here")
        _, result = self.call_json(self.start(), "/api/analyze", PNG)
        self.assertEqual(result["text"], "use <doc> tags like <doc>this</doc> here")

    def test_empty_doc_wrapper_is_502(self):
        self.upstream.routes[("POST", "/v1/chat/completions")] = chat_reply("<doc>  </doc>")
        status, _ = self.call_json(self.start(), "/api/analyze", PNG)
        self.assertEqual(status, 502)

    def test_only_think_block_is_502(self):
        self.upstream.routes[("POST", "/v1/chat/completions")] = chat_reply("<think>no answer</think>")
        status, _ = self.call_json(self.start(), "/api/analyze", PNG)
        self.assertEqual(status, 502)

    def test_multiline_text_is_preserved(self):
        self.upstream.routes[("POST", "/v1/chat/completions")] = chat_reply("line one\n  indented\n\nline four")
        _, result = self.call_json(self.start(), "/api/analyze", PNG)
        self.assertEqual(result["text"], "line one\n  indented\n\nline four")

    def test_image_exactly_at_limit_is_accepted(self):
        base = self.start(max_upload_bytes=len(PNG))
        status, _ = self.call_json(base, "/api/analyze", PNG)
        self.assertEqual(status, 200)

    def test_trailing_slash_in_backend_url_is_tolerated(self):
        base = self.start(backend_url=self.upstream.url + "/")
        status, _ = self.call_json(base, "/api/analyze", PNG)
        self.assertEqual(status, 200)


class ModelsEndpoint(AppTestCase):
    def test_only_vision_models_loaded_first(self):
        _, result = self.call_json(self.start(), "/api/models")
        self.assertEqual([m["id"] for m in result["models"]], ["vision-b", "vision-a"])
        self.assertEqual([m["loaded"] for m in result["models"]], [True, False])
        self.assertEqual(result["default"], "vision-b")

    def test_configured_default_is_reported(self):
        _, result = self.call_json(self.start(default_model="vision-a"), "/api/models")
        self.assertEqual(result["default"], "vision-a")

    def test_falls_back_to_openai_listing_when_v0_is_missing(self):
        self.upstream.routes.pop(("GET", "/api/v0/models"))
        self.upstream.routes[("GET", "/v1/models")] = (200, {"data": [{"id": "m1"}, {"id": "m2"}]})
        _, result = self.call_json(self.start(), "/api/models")
        self.assertEqual([m["id"] for m in result["models"]], ["m1", "m2"])
        self.assertEqual([m["loaded"] for m in result["models"]], [None, None])

    def test_unreachable_upstream_is_502(self):
        status, _ = self.call_json(self.start(backend_url="http://127.0.0.1:9"), "/api/models")
        self.assertEqual(status, 502)


class StaticAndRouting(AppTestCase):
    def test_index_page_is_served(self):
        status, raw = self.call(self.start(), "/")
        self.assertEqual(status, 200)
        self.assertIn(b"<html", raw.lower())

    def test_index_header_has_debug_button_top_right(self):
        _, raw = self.call(self.start(), "/")
        header = raw.decode().split("<header>", 1)[1].split("</header>", 1)[0]
        self.assertIn('<a class="button" href="/debug">Debug</a>', header)
        self.assertLess(header.index('href="/debug"'), header.index('id="theme"'))

    def test_debug_header_has_back_button(self):
        _, raw = self.call(self.start(), "/debug")
        header = raw.decode().split("<header>", 1)[1].split("</header>", 1)[0]
        self.assertIn('<a class="button" href="/">Back</a>', header)
        self.assertNotIn("Back to the app", raw.decode())

    def test_debug_statistics_wrap_long_model_names(self):
        _, raw = self.call(self.start(), "/debug")
        self.assertRegex(raw.decode(), r"#stats td\.model\s*\{[^}]*overflow-wrap:\s*anywhere")

    def test_both_pages_show_the_version_from_the_version_file(self):
        version = (Path(server.__file__).parent.parent / "VERSION").read_text().strip()
        base = self.start()
        for path in ("/", "/debug"):
            page = self.call(base, path)[1].decode()
            self.assertIn('<footer id="version">simple-ocr v%s</footer>' % version, page, path)
            self.assertNotIn(server.VERSION_PLACEHOLDER, page, path)

    def test_read_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "VERSION"
            self.assertEqual(server.read_version(path), "unknown")          # missing file
            path.write_text("0.2.3\n")
            self.assertEqual(server.read_version(path), "0.2.3")
            path.write_text("<script>alert(1)</script>")                     # never markup on the page
            self.assertEqual(server.read_version(path), "unknown")
            path.write_text("")
            self.assertEqual(server.read_version(path), "unknown")

    def test_unknown_path_is_404(self):
        status, _ = self.call_json(self.start(), "/etc/passwd")
        self.assertEqual(status, 404)

    def test_get_on_analyze_is_405(self):
        status, _ = self.call_json(self.start(), "/api/analyze")
        self.assertEqual(status, 405)



def truncated_body(handler):
    """Announce 1000 bytes, send 10, hang up: urllib raises IncompleteRead while reading."""
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", "1000")
    handler.end_headers()
    handler.wfile.write(b'{"choices"')
    handler.wfile.flush()
    handler.close_connection = True


class UpstreamMisbehaves(AppTestCase):
    """Every upstream failure must come back as a JSON error, never as a dropped connection."""

    def assert_json_error(self, status, result, expected_status):
        self.assertEqual(status, expected_status)
        self.assertIn("error", result)

    def test_truncated_upstream_body_is_502_json(self):
        self.upstream.routes[("POST", "/v1/chat/completions")] = (200, truncated_body)
        status, result = self.call_json(self.start(), "/api/analyze", PNG)
        self.assert_json_error(status, result, 502)

    def test_model_list_that_is_not_an_object_is_502(self):
        self.upstream.routes[("GET", "/api/v0/models")] = (200, [1, 2])
        self.upstream.routes[("GET", "/v1/models")] = (200, [1, 2])
        status, result = self.call_json(self.start(), "/api/models")
        self.assert_json_error(status, result, 502)

    def test_model_entries_without_id_are_skipped(self):
        self.upstream.routes[("GET", "/api/v0/models")] = (200, {"data": [
            {"type": "vlm", "state": "loaded"}, "junk", {"id": "ok", "type": "vlm", "state": "loaded"}]})
        status, result = self.call_json(self.start(), "/api/models")
        self.assertEqual(status, 200)
        self.assertEqual([m["id"] for m in result["models"]], ["ok"])

    def test_non_string_message_content_is_502(self):
        self.upstream.routes[("POST", "/v1/chat/completions")] = (
            200, {"choices": [{"message": {"content": [{"type": "text", "text": "x"}]}}]})
        status, result = self.call_json(self.start(), "/api/analyze", PNG)
        self.assert_json_error(status, result, 502)

    def test_unexpected_exception_is_500_json_and_server_survives(self):
        base = self.start()
        original = server.analyze
        server.analyze = lambda *args: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            status, result = self.call_json(base, "/api/analyze", PNG)
        finally:
            server.analyze = original
        self.assert_json_error(status, result, 500)
        self.assertIn("boom", result["error"])
        status, _ = self.call_json(base, "/api/analyze", PNG)   # still alive afterwards
        self.assertEqual(status, 200)


class ClientMisbehaves(AppTestCase):
    def raw_request(self, base, head, body=b"", half_close=True, read_response=True):
        import socket
        host, port = base.replace("http://", "").split(":")
        sock = socket.create_connection((host, int(port)), timeout=5)
        sock.sendall(head.encode() + body)
        if half_close:
            sock.shutdown(socket.SHUT_WR)
        if not read_response:
            return sock
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        sock.close()
        return b"".join(chunks)

    def test_truncated_upload_is_400_and_never_reaches_the_model(self):
        base = self.start()
        head = "POST /api/analyze HTTP/1.1\r\nHost: x\r\nContent-Length: 1000\r\n\r\n"
        response = self.raw_request(base, head, PNG)
        self.assertIn(b" 400 ", response.split(b"\r\n", 1)[0])
        self.assertEqual(self.chat_requests(), [])

    def test_client_hanging_up_does_not_raise_in_the_server(self):
        import time
        seen = []

        def slow_reply(handler):
            time.sleep(0.4)
            data = json.dumps({"choices": [{"message": {"content": "late"}}]}).encode()
            handler.send_response(200)
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)

        self.upstream.routes[("POST", "/v1/chat/completions")] = (200, slow_reply)
        app_url = self.start()
        self.apps[-1].handle_error = lambda request, client_address: seen.append(client_address)
        head = "POST /api/analyze HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n" % len(PNG)
        sock = self.raw_request(app_url, head, PNG, half_close=False, read_response=False)
        sock.close()                      # browser aborted, e.g. a newer request replaced this one
        time.sleep(1.0)                   # let the server finish and try to answer
        self.assertEqual(seen, [])


class CustomPrompt(AppTestCase):
    def sent_prompt(self):
        (_, _, sent), = self.chat_requests()
        return next(p for p in sent["messages"][-1]["content"] if p["type"] == "text")["text"]

    def analyze(self, base, **params):
        query = urllib.parse.urlencode(dict({"mode": "ocr", "model": "vision-b"}, **params))
        return self.call_json(base, "/api/analyze?" + query, PNG)

    def test_a_custom_prompt_replaces_the_default_verbatim(self):
        base = self.start()
        prompt = "Lies nur die Rechnungsnummer vor. Keine {language} Platzhalter, keine {} Klammern: äöü ß"
        status, body = self.analyze(base, prompt=prompt)
        self.assertEqual(status, 200, body)
        self.assertEqual(self.sent_prompt(), prompt)

    def test_an_empty_or_blank_prompt_means_the_default(self):
        base = self.start()
        self.assertEqual(self.analyze(base, prompt="   \n ")[0], 200)
        self.assertEqual(self.sent_prompt(), server.PROMPTS["ocr"])

    def test_a_too_long_prompt_is_refused(self):
        base = self.start()
        status, body = self.analyze(base, prompt="x" * (server.PROMPT_MAX_CHARS + 1))
        self.assertEqual(status, 400)
        self.assertIn(str(server.PROMPT_MAX_CHARS), body["error"])
        self.assertEqual(self.chat_requests(), [])

    def test_a_prompt_of_exactly_the_limit_is_accepted(self):
        base = self.start()
        self.assertEqual(self.analyze(base, prompt="y" * server.PROMPT_MAX_CHARS)[0], 200)

    def test_switched_off_the_server_refuses_a_custom_prompt(self):
        base = self.start(custom_prompt=False)
        status, body = self.analyze(base, prompt="Say hello")
        self.assertEqual(status, 403)
        self.assertIn("Custom prompts are disabled", body["error"])
        self.assertEqual(self.chat_requests(), [])

    def test_switched_off_the_default_prompt_still_works(self):
        base = self.start(custom_prompt=False)
        self.assertEqual(self.analyze(base)[0], 200)
        self.assertEqual(self.sent_prompt(), server.PROMPTS["ocr"])

    def test_settings_report_the_switches_and_the_default_prompts(self):
        status, body = self.call_json(self.start(default_language="French"), "/api/settings")
        self.assertEqual(status, 200)
        self.assertEqual((body["custom_prompt"], body["debug_page"], body["batch_page"]), (True, True, True))
        self.assertEqual(body["prompts"]["ocr"], server.PROMPTS["ocr"])
        self.assertIn("French", body["prompts"]["describe"])
        self.assertEqual(body["prompt_max_chars"], server.PROMPT_MAX_CHARS)
        _, off = self.call_json(self.start(custom_prompt=False, debug_page=False, batch_page=False), "/api/settings")
        self.assertEqual((off["custom_prompt"], off["debug_page"], off["batch_page"]), (False, False, False))

    def test_the_prompt_editor_is_only_in_the_page_when_switched_on(self):
        _, on = self.call(self.start(), "/")
        _, off = self.call(self.start(custom_prompt=False), "/")
        self.assertIn(b'id="prompt"', on)
        self.assertNotIn(b'id="prompt"', off)
        self.assertNotIn(b"<!--PROMPT", on + off)


class UpstreamSafety(AppTestCase):
    def test_redirects_are_not_followed_so_the_key_stays_with_the_configured_server(self):
        other = FakeUpstream()
        self.addCleanup(other.close)
        other.routes[("GET", "/api/v0/models")] = (200, V0_MODELS)

        def redirect(handler):
            handler.send_response(302)
            handler.send_header("Location", other.url + "/api/v0/models")
            handler.send_header("Content-Length", "0")
            handler.end_headers()
        self.upstream.routes[("GET", "/api/v1/models")] = (302, redirect)
        self.upstream.routes[("GET", "/api/v0/models")] = (302, redirect)
        self.upstream.routes[("GET", "/v1/models")] = (302, redirect)
        status, body = self.call_json(self.start(api_key="sk-SECRET"), "/api/models")
        self.assertEqual(status, 502, body)
        self.assertIn("redirect", body["error"])
        self.assertEqual(other.requests, [], "the other host must not be contacted")

    def test_the_backend_url_is_not_sent_to_the_browser(self):
        base = self.start(backend_url="http://127.0.0.1:9/private-path")
        status, raw = self.call(base, "/api/models")
        self.assertEqual(status, 502)
        self.assertIn(b"is not reachable", raw)
        self.assertNotIn(b"private-path", raw)
        self.assertNotIn(b"127.0.0.1:9", raw)

    def test_a_certificate_for_another_name_does_not_name_the_host_in_the_browser(self):
        import ssl
        reason = ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
                                                 "IP address mismatch, certificate is not valid for '203.0.113.77'")
        with mock.patch.object(backends, "_open", side_effect=urllib.error.URLError(reason)):
            status, raw = self.call(self.start(), "/api/models")
        self.assertEqual(status, 502)
        self.assertIn(b"CERTIFICATE_VERIFY_FAILED", raw)
        self.assertIn(b"LMSTUDIO_TLS_VERIFY", raw)
        self.assertNotIn(b"203.0.113.77", raw)

    def test_other_connection_errors_do_not_name_the_host_either(self):
        import socket
        for reason in (socket.gaierror(-2, "Name or service not known for secret-host.example"),
                       ConnectionRefusedError(111, "Connection refused by secret-host.example")):
            with mock.patch.object(backends, "_open", side_effect=urllib.error.URLError(reason)):
                status, raw = self.call(self.start(), "/api/models")
            self.assertEqual(status, 502)
            self.assertNotIn(b"secret-host", raw)

    def test_a_refused_upload_is_read_before_the_answer(self):
        base = self.start(debug_page=False)
        big = PNG + b"\0" * (6 * 1024 * 1024)
        status, raw = self.call(base, "/api/debug/runs?mode=ocr&all=1", big)
        self.assertEqual(status, 403)
        self.assertIn(b"debug page is disabled", raw)


class LoadedCardHeading(unittest.TestCase):
    def test_title_and_total_are_the_only_two_flex_items(self):
        # h2 is display:flex with space-between: a third child spreads "Loaded on" and the name apart
        from html.parser import HTMLParser

        class Children(HTMLParser):
            def __init__(self):
                super().__init__()
                self.depth, self.inside, self.items = 0, False, []

            def handle_starttag(self, tag, attrs):
                if self.inside:
                    if self.depth == 0:
                        self.items.append(tag)
                    self.depth += 1
                elif tag == "h2" and self.after_card:
                    self.inside = True

            def handle_endtag(self, tag):
                if self.inside:
                    if self.depth == 0:
                        self.inside, self.after_card = False, False
                    else:
                        self.depth -= 1

            def handle_data(self, data):
                if self.inside and self.depth == 0 and data.strip():
                    self.items.append("text")

        parser = Children()
        html = server.INDEX_PATH.read_text(encoding="utf-8")
        parser.after_card = True   # fed from the loaded card on, so its first h2 is the one
        parser.feed(html[html.index('id="loaded-card"'):])
        self.assertEqual(parser.items, ["span", "small"])


class DebugPageSwitch(AppTestCase):
    def test_switched_off_every_debug_route_is_refused_on_the_server(self):
        base = self.start(debug_page=False, data_dir="")
        for method, path in (("GET", "/debug"), ("GET", "/api/debug/saved"), ("GET", "/api/debug/runs/latest"),
                             ("POST", "/api/debug/runs?mode=ocr&all=1"), ("POST", "/api/debug/runs/0123456789ab/cancel")):
            status, raw = self.call(base, path, PNG if method == "POST" else None)
            self.assertEqual(status, 403, path)
            self.assertIn(b"debug page is disabled", raw, path)
        self.assertEqual(self.upstream.requests, [], "nothing may reach the model server")

    def test_switched_off_the_button_is_not_in_the_page(self):
        _, page = self.call(self.start(debug_page=False), "/")
        self.assertNotIn(b'href="/debug"', page)
        self.assertNotIn(b"<!--DEBUG", page)

    def test_switched_on_the_page_and_button_are_there(self):
        base = self.start()
        _, page = self.call(base, "/")
        self.assertIn(b'href="/debug"', page)
        self.assertEqual(self.call(base, "/debug")[0], 200)
        self.assertEqual(self.call(base, "/api/debug/saved")[0], 200)

    def test_the_main_page_keeps_working_without_debug(self):
        status, body = self.call_json(self.start(debug_page=False), "/api/analyze?mode=ocr&model=vision-b", PNG)
        self.assertEqual(status, 200, body)


class ConfigFromEnv(unittest.TestCase):
    def test_defaults_and_overrides(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {}, clear=True):
            config = server.Config.from_env()
        self.assertEqual(config.default_model, "")
        self.assertEqual(config.max_upload_bytes, 20 * 1024 * 1024)
        env = {"LMSTUDIO_URL": "http://h:1", "DEFAULT_MODEL": "m", "MAX_UPLOAD_MB": "0.5",
               "UPSTREAM_TIMEOUT": "7", "DESCRIBE_LANGUAGE": "French"}
        with mock.patch.dict(os.environ, env, clear=True):
            config = server.Config.from_env()
        self.assertEqual((config.backend_url, config.default_model, config.max_upload_bytes,
                          config.timeout, config.default_language),
                         ("http://h:1", "m", 524288, 7.0, "French"))



GB = 1000 ** 3
V1_MODELS = {"models": [
    {"key": "text-only", "type": "llm", "size_bytes": 4 * GB, "params_string": "8B",
     "quantization": {"name": "Q4_K_M"}, "capabilities": {"vision": False},
     "loaded_instances": [{"id": "text-only"}]},
    {"key": "vision-a", "type": "llm", "size_bytes": 2 * GB, "params_string": "2B",
     "quantization": {"name": "Q4_K_M"}, "capabilities": {"vision": True}, "loaded_instances": []},
    {"key": "vision-b", "type": "llm", "size_bytes": 30 * GB, "params_string": "27B",
     "quantization": {"name": "8bit"}, "capabilities": {"vision": True},
     "loaded_instances": [{"id": "vision-b"}]},
    {"key": "embed", "type": "embedding", "size_bytes": GB, "params_string": "0.6B",
     "quantization": {"name": "Q8_0"}, "loaded_instances": []},
]}


class ModelListingWithSizes(AppTestCase):
    def setUp(self):
        super().setUp()
        self.upstream.routes[("GET", "/api/v1/models")] = (200, V1_MODELS)

    def test_vision_models_carry_size_params_and_quantization(self):
        _, result = self.call_json(self.start(), "/api/models")
        by_id = {m["id"]: m for m in result["models"]}
        self.assertEqual(list(by_id), ["vision-b", "vision-a"])      # loaded first, text-only excluded
        self.assertEqual(by_id["vision-b"], {"id": "vision-b", "loaded": True, "size_bytes": 30 * GB,
                                             "params": "27B", "quantization": "8bit"})
        self.assertEqual(by_id["vision-a"]["size_bytes"], 2 * GB)
        self.assertFalse(by_id["vision-a"]["loaded"])

    def test_loaded_now_lists_every_loaded_model_including_non_vision(self):
        _, result = self.call_json(self.start(), "/api/models")
        self.assertEqual(
            sorted(result["loaded_now"], key=lambda m: m["id"]),
            [{"id": "text-only", "size_bytes": 4 * GB, "params": "8B", "quantization": "Q4_K_M", "kind": "text"},
             {"id": "vision-b", "size_bytes": 30 * GB, "params": "27B", "quantization": "8bit", "kind": "vision"}])

    def test_loaded_embedding_model_is_marked_as_embedding(self):
        listing = {"models": [
            {"key": "e1", "type": "embedding", "size_bytes": GB, "loaded_instances": [{"id": "e1"}]},
            {"key": "jina-embeddings-x", "type": "llm", "capabilities": {"vision": True},
             "loaded_instances": [{"id": "jina-embeddings-x"}]}]}
        self.upstream.routes[("GET", "/api/v1/models")] = (200, listing)
        _, result = self.call_json(self.start(), "/api/models")
        self.assertEqual({m["id"]: m["kind"] for m in result["loaded_now"]},
                         {"e1": "embedding", "jina-embeddings-x": "embedding"})
        self.assertEqual(result["models"], [])

    def test_default_is_still_first_loaded_vision_model(self):
        _, result = self.call_json(self.start(), "/api/models")
        self.assertEqual(result["default"], "vision-b")

    def test_falls_back_to_v0_without_sizes_when_v1_is_missing(self):
        self.upstream.routes.pop(("GET", "/api/v1/models"))
        _, result = self.call_json(self.start(), "/api/models")
        self.assertEqual([m["id"] for m in result["models"]], ["vision-b", "vision-a"])
        self.assertIsNone(result["models"][0]["size_bytes"])
        self.assertEqual([m["id"] for m in result["loaded_now"]], ["text-only", "vision-b"])
        self.assertEqual([m["kind"] for m in result["loaded_now"]], ["text", "vision"])
        self.assertIsNone(result["loaded_now"][0]["size_bytes"])

    def test_entries_with_missing_fields_do_not_break_the_listing(self):
        self.upstream.routes[("GET", "/api/v1/models")] = (200, {"models": [
            {"key": "bare", "capabilities": {"vision": True}}, {"key": "x"}, 7]})
        _, result = self.call_json(self.start(), "/api/models")
        self.assertEqual(result["models"], [{"id": "bare", "loaded": False, "size_bytes": None,
                                             "params": None, "quantization": None}])
        self.assertEqual(result["loaded_now"], [])

    def test_embedding_models_are_not_offered_even_with_vision(self):
        self.upstream.routes[("GET", "/api/v1/models")] = (200, {"models": [
            {"key": "jina-embeddings-v5-omni-nano-mlx", "type": "llm", "capabilities": {"vision": True}},
            {"key": "clip-thing", "type": "embedding", "capabilities": {"vision": True}},
            {"key": "real-vision", "type": "llm", "capabilities": {"vision": True}}]})
        _, result = self.call_json(self.start(), "/api/models")
        self.assertEqual([m["id"] for m in result["models"]], ["real-vision"])

    def test_embedding_models_are_not_offered_by_the_v0_fallback_either(self):
        self.upstream.routes.pop(("GET", "/api/v1/models"))   # force the v0 path
        self.upstream.routes[("GET", "/api/v0/models")] = (200, {"data": [
            {"id": "jina-embeddings-v5-omni-small-mlx", "type": "vlm", "state": "not-loaded"},
            {"id": "real-vision", "type": "vlm", "state": "not-loaded"}]})
        _, result = self.call_json(self.start(), "/api/models")
        self.assertEqual([m["id"] for m in result["models"]], ["real-vision"])

    def test_default_with_nothing_loaded_is_the_smallest_vision_model(self):
        listing = {"models": [
            {"key": "big", "type": "llm", "size_bytes": 30 * GB, "capabilities": {"vision": True}, "loaded_instances": []},
            {"key": "small", "type": "llm", "size_bytes": 2 * GB, "capabilities": {"vision": True}, "loaded_instances": []},
            {"key": "unknown-size", "type": "llm", "capabilities": {"vision": True}, "loaded_instances": []}]}
        self.upstream.routes[("GET", "/api/v1/models")] = (200, listing)
        _, result = self.call_json(self.start(), "/api/models")
        self.assertEqual(result["default"], "small")

    def test_a_loaded_model_still_beats_a_smaller_one(self):
        listing = {"models": [
            {"key": "small", "type": "llm", "size_bytes": 2 * GB, "capabilities": {"vision": True}, "loaded_instances": []},
            {"key": "big-loaded", "type": "llm", "size_bytes": 30 * GB, "capabilities": {"vision": True},
             "loaded_instances": [{"id": "big-loaded"}]}]}
        self.upstream.routes[("GET", "/api/v1/models")] = (200, listing)
        _, result = self.call_json(self.start(), "/api/models")
        self.assertEqual(result["default"], "big-loaded")

    def test_nothing_loaded_gives_empty_loaded_now(self):
        for m in V1_MODELS["models"]:
            pass
        quiet = {"models": [dict(m, loaded_instances=[]) for m in V1_MODELS["models"]]}
        self.upstream.routes[("GET", "/api/v1/models")] = (200, quiet)
        _, result = self.call_json(self.start(), "/api/models")
        self.assertEqual(result["loaded_now"], [])
        self.assertEqual(result["default"], "vision-a")        # first vision model, none loaded


class ApiKey(AppTestCase):
    def test_key_is_sent_as_bearer_token_on_every_upstream_call(self):
        base = self.start(api_key="secret-token")
        self.call_json(base, "/api/models")
        self.call_json(base, "/api/analyze", PNG)
        self.assertGreaterEqual(len(self.upstream.headers), 2)
        for headers in self.upstream.headers:
            self.assertEqual(headers.get("Authorization"), "Bearer secret-token")

    def test_no_authorization_header_without_a_key(self):
        base = self.start()
        self.call_json(base, "/api/models")
        self.assertTrue(all("Authorization" not in h for h in self.upstream.headers))

    def test_key_never_appears_in_answers_or_repr(self):
        self.upstream.routes[("POST", "/v1/chat/completions")] = (401, {"error": "invalid api key"})
        base = self.start(api_key="secret-token")
        status, raw = self.call(base, "/api/analyze", PNG)
        self.assertEqual(status, 502)
        self.assertNotIn(b"secret-token", raw)
        self.assertNotIn("secret-token", repr(server.Config(api_key="secret-token")))


class ConfigHostPortAndEnvFile(unittest.TestCase):
    def env(self, **values):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, values, clear=True):
            return server.Config.from_env()

    def test_host_and_port_build_the_url(self):
        self.assertEqual(self.env(LMSTUDIO_HOST="192.0.2.5", LMSTUDIO_PORT="12345").backend_url, "http://192.0.2.5:12345")

    def test_port_defaults_to_1234(self):
        self.assertEqual(self.env(LMSTUDIO_HOST="192.0.2.5").backend_url, "http://192.0.2.5:1234")

    def test_full_url_wins_over_host_and_port(self):
        config = self.env(LMSTUDIO_URL="https://lm.example:8443", LMSTUDIO_HOST="192.0.2.5", LMSTUDIO_PORT="1")
        self.assertEqual(config.backend_url, "https://lm.example:8443")

    def test_ipv6_host_is_bracketed(self):
        self.assertEqual(self.env(LMSTUDIO_HOST="fe80::1", LMSTUDIO_PORT="1234").backend_url, "http://[fe80::1]:1234")

    def test_scheme_in_host_is_rejected_with_a_hint(self):
        with self.assertRaises(ValueError) as caught:
            self.env(LMSTUDIO_HOST="http://192.0.2.5")
        self.assertIn("LMSTUDIO_URL", str(caught.exception))

    def test_non_numeric_port_is_rejected(self):
        with self.assertRaises(ValueError):
            self.env(LMSTUDIO_HOST="192.0.2.5", LMSTUDIO_PORT="abc")

    def test_api_key_is_read_and_blank_means_none(self):
        self.assertEqual(self.env(LMSTUDIO_API_KEY="k").api_key, "k")
        self.assertEqual(self.env(LMSTUDIO_API_KEY="  ").api_key, "")

    def test_tls_is_verified_unless_the_active_server_switches_it_off(self):
        self.assertTrue(self.env().tls_verify)
        self.assertTrue(self.env(OLLAMA_TLS_VERIFY="").tls_verify)
        for value in ("false", "FALSE", " 0 ", "no", "Off"):
            self.assertFalse(self.env(BACKEND="ollama", OLLAMA_TLS_VERIFY=value).tls_verify, value)
        for value in ("true", "1", "yes", "on"):
            self.assertTrue(self.env(BACKEND="ollama", OLLAMA_TLS_VERIFY=value).tls_verify, value)

    def test_tls_flag_of_another_server_does_not_count(self):
        self.assertTrue(self.env(BACKEND="ollama", LMSTUDIO_TLS_VERIFY="false").tls_verify)
        self.assertFalse(self.env(BACKEND="vllm", VLLM_TLS_VERIFY="false").tls_verify)

    def test_unclear_tls_flag_is_rejected_with_its_name(self):
        with self.assertRaises(ValueError) as caught:
            self.env(BACKEND="llamacpp", LLAMACPP_TLS_VERIFY="maybe")
        self.assertIn("LLAMACPP_TLS_VERIFY", str(caught.exception))

    def test_startup_line_warns_when_the_certificate_is_not_checked(self):
        line = server.startup_line(server.Config(backend="ollama", backend_url="https://ollama.example/ollama-api/",
                                                 tls_verify=False), "0.0.0.0", 8080)
        self.assertIn("https://ollama.example/ollama-api/", line)
        self.assertIn("certificate is NOT verified", line)
        self.assertNotIn("NOT verified", server.startup_line(server.Config(), "0.0.0.0", 8080))

    def test_a_url_without_http_or_https_is_rejected_at_start(self):
        for value in ("secret-lm.example:1234", "192.0.2.77:1234", "secret-lm.example", "ftp://lm.example"):
            with self.assertRaises(ValueError) as caught:
                self.env(LMSTUDIO_URL=value)
            self.assertIn("LMSTUDIO_URL", str(caught.exception))
            self.assertNotIn(value, str(caught.exception), "the value itself is not echoed")
        self.assertEqual(self.env(LMSTUDIO_URL=" HTTPS://lm.example ").backend_url, "HTTPS://lm.example")

    def test_custom_prompt_and_debug_page_are_on_unless_switched_off(self):
        config = self.env()
        self.assertEqual((config.custom_prompt, config.debug_page), (True, True))
        config = self.env(CUSTOM_PROMPT="false", DEBUG_PAGE="0")
        self.assertEqual((config.custom_prompt, config.debug_page), (False, False))
        config = self.env(CUSTOM_PROMPT=" Yes ", DEBUG_PAGE="on")
        self.assertEqual((config.custom_prompt, config.debug_page), (True, True))

    def test_unclear_switch_values_are_rejected_with_their_name(self):
        for name in ("CUSTOM_PROMPT", "DEBUG_PAGE"):
            with self.assertRaises(ValueError) as caught:
                self.env(**{name: "sometimes"})
            self.assertIn(name, str(caught.exception))

    def write_env(self, text):
        import tempfile
        handle = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
        handle.write(text)
        handle.close()
        self.addCleanup(__import__("os").unlink, handle.name)
        return handle.name

    def test_env_file_is_parsed(self):
        path = self.write_env("# comment\n\nLMSTUDIO_HOST=192.0.2.1\nLMSTUDIO_API_KEY=\"abc def\"\nDEFAULT_MODEL = m1\n")
        self.assertEqual(server.parse_env_file(path),
                         {"LMSTUDIO_HOST": "192.0.2.1", "LMSTUDIO_API_KEY": "abc def", "DEFAULT_MODEL": "m1"})

    def test_missing_env_file_is_empty(self):
        self.assertEqual(server.parse_env_file("/nonexistent/.env"), {})

    def test_malformed_line_names_the_line_but_not_its_content(self):
        path = self.write_env("OK=1\nthis-is-a-secret-without-equals\n")
        with self.assertRaises(ValueError) as caught:
            server.parse_env_file(path)
        self.assertIn("line 2", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))

    def test_load_env_file_does_not_override_the_real_environment(self):
        import os
        from unittest import mock
        path = self.write_env("LMSTUDIO_HOST=from-file\nDEFAULT_MODEL=from-file\n")
        with mock.patch.dict(os.environ, {"LMSTUDIO_HOST": "from-env"}, clear=True):
            server.load_env_file(path)
            self.assertEqual(os.environ["LMSTUDIO_HOST"], "from-env")
            self.assertEqual(os.environ["DEFAULT_MODEL"], "from-file")


if __name__ == "__main__":
    unittest.main()
