"""Batch mode: many images or ZIP archives through one model, reports as Markdown, HTML or PDF."""
import base64
import io
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from unittest import mock

from app import backends, batch, report, server
from tests.test_backends import GB, FakeOllama

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff" + b"\x00" * 32


def real_png(width=3, height=2):
    """A valid PNG made with the standard library (the other test images are only headers)."""
    import struct
    import zlib

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def image(marker, kind=PNG):
    """A tiny 'image' whose marker the fake model reads back."""
    return kind + b"#" + marker.encode()


def make_zip(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return buffer.getvalue()


def multipart(files, fields=()):
    boundary = "----simpleocr" + uuid.uuid4().hex
    body = io.BytesIO()
    for name, value in fields:
        body.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n" % (boundary, name, value)).encode())
    for filename, data in files:
        body.write(("--%s\r\nContent-Disposition: form-data; name=\"files\"; filename=\"%s\"\r\n"
                    "Content-Type: application/octet-stream\r\n\r\n" % (boundary, filename)).encode())
        body.write(data + b"\r\n")
    body.write(("--%s--\r\n" % boundary).encode())
    return "multipart/form-data; boundary=" + boundary, body.getvalue()


class ReadingFake(FakeOllama):
    """Answers with the marker inside the image; a marker starting with FAIL makes the request fail."""

    delay = 0.0

    def route(self, method, path, body):
        if (method, path) == ("POST", "/api/chat"):
            raw = base64.b64decode(body["messages"][0]["images"][0])
            marker = raw.split(b"#", 1)[1].decode() if b"#" in raw else "?"
            with self.lock:
                self.events.append(("chat", body["model"], marker, body["messages"][0]["content"]))
            time.sleep(self.delay)
            if marker.startswith("FAIL"):
                return 500, {"error": "the model crashed on " + marker}
            return 200, {"model": body["model"], "message": {"role": "assistant", "content": "Text of " + marker},
                         "done": True}
        return super().route(method, path, body)


class Collect(unittest.TestCase):
    LIMITS = dict(max_image_bytes=1000, max_total_bytes=100000)

    def test_images_are_kept_in_natural_order_and_other_files_are_skipped(self):
        images, skipped = batch.collect([("img10.png", PNG), ("img2.jpg", JPEG), ("notes.txt", b"hello")], **self.LIMITS)
        self.assertEqual([name for name, _ in images], ["img2.jpg", "img10.png"])
        self.assertEqual(skipped, [("notes.txt", "not a PNG, JPEG or WebP image")])

    def test_a_zip_is_opened_with_its_folders_and_without_system_files(self):
        archive = make_zip([("scan/b.png", PNG), ("a.png", JPEG), ("__MACOSX/scan/._b.png", b"junk"),
                            (".DS_Store", b"junk"), ("scan/readme.txt", b"text"), ("inner.zip", make_zip([("x.png", PNG)]))])
        images, skipped = batch.collect([("pages.zip", archive)], **self.LIMITS)
        self.assertEqual([name for name, _ in images], ["pages.zip/a.png", "pages.zip/scan/b.png"])
        self.assertEqual(dict(skipped), {"pages.zip/scan/readme.txt": "not a PNG, JPEG or WebP image",
                                         "pages.zip/inner.zip": "archives inside archives are not opened"})

    def test_entries_above_the_image_limit_or_beyond_the_total_are_skipped(self):
        archive = make_zip([("big.png", PNG + b"\0" * 2000), ("ok.png", PNG), ("late.png", PNG)])
        images, skipped = batch.collect([("a.zip", archive)], max_image_bytes=1000, max_total_bytes=len(PNG) + 10)
        self.assertEqual([name for name, _ in images], ["a.zip/ok.png"])
        self.assertIn(("a.zip/big.png", "larger than the 0 MB image limit"), skipped)
        self.assertIn(("a.zip/late.png", "the batch is full"), skipped)

    def test_a_damaged_zip_is_skipped_not_fatal(self):
        archive = make_zip([("a.png", PNG)])
        images, skipped = batch.collect([("broken.zip", archive[:-30]), ("ok.png", PNG)], **self.LIMITS)
        self.assertEqual([name for name, _ in images], ["ok.png"])
        self.assertEqual([name for name, _ in skipped], ["broken.zip"])

    def test_more_than_the_file_limit_is_skipped(self):
        with mock.patch.object(batch, "MAX_FILES", 2):
            images, skipped = batch.collect([("%d.png" % i, PNG) for i in range(4)], **self.LIMITS)
        self.assertEqual(len(images), 2)
        self.assertEqual([reason for _, reason in skipped], ["more than 2 images"] * 2)

    def test_nothing_at_all(self):
        self.assertEqual(batch.collect([], **self.LIMITS), ([], []))


class Multipart(unittest.TestCase):
    def test_files_and_their_base_names_are_read(self):
        content_type, body = multipart([("scans/a.png", PNG), ("dir/b.zip", b"PK")], fields=[("note", "x")])
        self.assertEqual(batch.parse_upload(content_type, body), [("a.png", PNG), ("b.zip", b"PK")])

    def test_wrong_or_empty_uploads_are_refused(self):
        with self.assertRaises(backends.ApiError) as caught:
            batch.parse_upload("application/octet-stream", PNG)
        self.assertEqual(caught.exception.status, 415)
        content_type, body = multipart([], fields=[("note", "x")])
        with self.assertRaises(backends.ApiError) as caught:
            batch.parse_upload(content_type, body)
        self.assertEqual(caught.exception.status, 400)


class BatchApi(unittest.TestCase):
    def setUp(self):
        self.fake = ReadingFake({"granite3.2-vision:2b": {"caps": ["vision"], "size": 2 * GB},
                                 "llava:7b": {"caps": ["vision"], "size": 4 * GB},
                                 "qwen3:8b": {"caps": ["completion"], "size": 5 * GB}}, loaded=["llava:7b"])
        self.addCleanup(self.fake.close)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def start(self, **config):
        settings = dict(backend="ollama", backend_url=self.fake.url, timeout=5, allow_loading=True, data_dir=self.tmp.name)
        settings.update(config)
        app = server.make_server(server.Config(**settings), "127.0.0.1", 0)
        threading.Thread(target=app.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        self.addCleanup(app.server_close)
        self.addCleanup(app.shutdown)
        self.app = app
        return "http://127.0.0.1:%d" % app.server_address[1]

    def call(self, base, path, body=None, content_type=None, method=None):
        request = urllib.request.Request(base + path, data=body, method=method or ("POST" if body is not None else "GET"))
        if content_type:
            request.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.read(), response.headers
        except urllib.error.HTTPError as error:
            return error.code, error.read(), error.headers

    def call_json(self, *args, **kwargs):
        status, raw, _ = self.call(*args, **kwargs)
        return status, json.loads(raw)

    def submit(self, base, files, **params):
        content_type, body = multipart(files)
        query = urllib.parse.urlencode(dict({"mode": "ocr"}, **params))
        return self.call_json(base, "/api/batch?" + query, body, content_type)

    def wait(self, base, job_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            _, state = self.call_json(base, "/api/batch/" + job_id)
            if state["status"] != "running":
                return state
            time.sleep(0.02)
        self.fail("batch did not finish")

    def chats(self):
        return [e[2] for e in self.fake.events if e[0] == "chat"]

    def test_images_and_a_zip_are_analyzed_in_name_order(self):
        base = self.start()
        archive = make_zip([("p2.png", image("p2")), ("p10.png", image("p10")), ("x.txt", b"no")])
        status, started = self.submit(base, [("b.png", image("b")), ("a.jpg", image("a", JPEG)), ("pages.zip", archive)])
        self.assertEqual(status, 200, started)
        self.assertEqual(started["total"], 4)
        self.assertEqual(started["skipped"], [{"name": "pages.zip/x.txt", "reason": "not a PNG, JPEG or WebP image"}])
        state = self.wait(base, started["id"])
        self.assertEqual(state["status"], "done")
        self.assertEqual([(i["name"], i["status"], i["text"]) for i in state["items"]],
                         [("a.jpg", "ok", "Text of a"), ("b.png", "ok", "Text of b"),
                          ("pages.zip/p2.png", "ok", "Text of p2"), ("pages.zip/p10.png", "ok", "Text of p10")])
        self.assertEqual(state["model"], "llava:7b")      # the loaded vision model
        _, listing = self.call_json(base, "/api/batch")
        self.assertEqual([j["id"] for j in listing["batches"]], [started["id"]])

    def test_a_failing_image_is_recorded_and_the_batch_goes_on(self):
        base = self.start()
        _, started = self.submit(base, [("1.png", image("one")), ("2.png", image("FAIL-two")), ("3.png", image("three"))])
        state = self.wait(base, started["id"])
        self.assertEqual(state["status"], "done")
        self.assertEqual([i["status"] for i in state["items"]], ["ok", "error", "ok"])
        self.assertIn("the model crashed on FAIL-two", state["items"][1]["error"])

    def test_model_mode_and_custom_prompt_are_used(self):
        base = self.start()
        _, started = self.submit(base, [("1.png", image("one"))], mode="describe", model="granite3.2-vision",
                                 prompt="Nur die Rechnungsnummer")
        state = self.wait(base, started["id"])
        self.assertEqual((state["model"], state["mode"], state["prompt"]), ("granite3.2-vision:2b", "describe", "Nur die Rechnungsnummer"))
        self.assertEqual([(e[1], e[3]) for e in self.fake.events if e[0] == "chat"], [("granite3.2-vision:2b", "Nur die Rechnungsnummer")])

    def test_wrong_requests_are_refused_before_a_batch_starts(self):
        base = self.start()
        self.assertEqual(self.submit(base, [("1.png", image("one"))], model="qwen3")[0], 400)       # a text model
        self.assertEqual(self.submit(base, [("1.png", image("one"))], mode="poem")[0], 400)
        status, body = self.submit(base, [("notes.txt", b"hello")])
        self.assertEqual(status, 400)
        self.assertIn("No image", body["error"])
        self.assertEqual(self.call_json(base, "/api/batch?mode=ocr", PNG, "image/png")[0], 415)
        self.assertEqual(self.chats(), [])

    def test_custom_prompt_switched_off_is_refused(self):
        base = self.start(custom_prompt=False)
        status, body = self.submit(base, [("1.png", image("one"))], prompt="hello")
        self.assertEqual(status, 403)
        self.assertEqual(self.chats(), [])

    def test_the_upload_limit_applies(self):
        base = self.start(max_batch_bytes=2000)
        status, body = self.submit(base, [("1.png", image("one") + b"\0" * 3000)])
        self.assertEqual(status, 413)

    def test_batch_benchmark_and_a_second_batch_exclude_each_other(self):
        self.fake.delay = 0.3
        base = self.start()
        _, started = self.submit(base, [("1.png", image("one")), ("2.png", image("two"))])
        self.assertEqual(self.submit(base, [("3.png", image("three"))])[0], 409)
        status, body = self.call_json(base, "/api/debug/runs?mode=ocr&model=llava:7b", PNG)
        self.assertEqual(status, 409)
        self.wait(base, started["id"])
        self.assertEqual(self.call_json(base, "/api/debug/runs?mode=ocr&model=llava:7b", PNG)[0], 200)
        deadline = time.monotonic() + 5
        while self.app.RequestHandlerClass.runs.active() and time.monotonic() < deadline:
            time.sleep(0.02)

    def test_a_batch_cannot_start_while_a_benchmark_runs(self):
        self.fake.delay = 0.3
        base = self.start()
        self.assertEqual(self.call_json(base, "/api/debug/runs?mode=ocr&model=llava:7b", PNG)[0], 200)
        status, body = self.submit(base, [("1.png", image("one"))])
        self.assertEqual(status, 409)
        self.assertIn("benchmark", body["error"])

    def test_cancel_stops_before_the_next_image(self):
        self.fake.delay = 0.3
        base = self.start()
        _, started = self.submit(base, [("%d.png" % i, image(str(i))) for i in range(4)])
        time.sleep(0.1)
        self.call(base, "/api/batch/%s/cancel" % started["id"], b"")
        state = self.wait(base, started["id"])
        self.assertEqual(state["status"], "cancelled")
        self.assertEqual([i["status"] for i in state["items"]], ["ok", "cancelled", "cancelled", "cancelled"])

    def test_switched_off_every_batch_route_is_refused(self):
        base = self.start(batch_page=False)
        _, page = self.call(base, "/")[:2]
        self.assertNotIn(b'href="/batch"', page)
        content_type, body = multipart([("1.png", image("one"))])
        for method, path, data in (("GET", "/batch", None), ("GET", "/api/batch", None),
                                   ("POST", "/api/batch?mode=ocr", body), ("GET", "/api/batch/0123456789ab", None),
                                   ("GET", "/api/batch/0123456789ab/report.md", None)):
            status, raw, _ = self.call(base, path, data, content_type if data else None)
            self.assertEqual(status, 403, path)
            self.assertIn(b"batch page is disabled", raw)
        self.assertEqual(self.chats(), [])

    def test_switched_on_the_page_and_button_are_there(self):
        base = self.start()
        self.assertIn(b'href="/batch"', self.call(base, "/")[1])
        status, page, _ = self.call(base, "/batch")
        self.assertEqual(status, 200)
        self.assertIn(b"simple-ocr", page)

    def test_reports_are_downloads_and_unknown_ones_are_404(self):
        base = self.start()
        _, started = self.submit(base, [("1.png", image("one"))])
        self.wait(base, started["id"])
        for ext, kind in (("md", "text/markdown"), ("html", "text/html")):
            status, raw, headers = self.call(base, "/api/batch/%s/report.%s" % (started["id"], ext))
            self.assertEqual(status, 200)
            self.assertIn(kind, headers["Content-Type"])
            self.assertIn('attachment; filename="simple-ocr-batch-%s.%s"' % (started["id"], ext), headers["Content-Disposition"])
            self.assertIn(b"Text of one", raw)
        self.assertEqual(self.call(base, "/api/batch/%s/report.doc" % started["id"])[0], 404)
        self.assertEqual(self.call(base, "/api/batch/0123456789ab/report.md")[0], 404)
        self.assertEqual(self.call(base, "/api/batch/..%2F..%2Fetc/report.md")[0], 404)

    def test_no_image_stays_in_memory_or_on_disk(self):
        base = self.start()
        _, started = self.submit(base, [("1.png", image("one")), ("2.png", image("two"))])
        self.wait(base, started["id"])
        job = self.app.RequestHandlerClass.batches.get(started["id"])
        self.assertEqual(job.images, [None, None])
        self.assertEqual(job.thumbnails, {})
        written = [os.path.join(root, f) for root, _, files in os.walk(self.tmp.name) for f in files]
        self.assertEqual(written, [])

    def test_embedded_images_are_kept_only_as_thumbnails(self):
        base = self.start()
        _, started = self.submit(base, [("1.png", image("one", real_png())), ("2.png", image("broken"))], images="1")
        self.wait(base, started["id"])
        job = self.app.RequestHandlerClass.batches.get(started["id"])
        self.assertEqual(job.images, [None, None])
        self.assertEqual(set(job.thumbnails), {0}, "a picture that cannot be read only loses its preview")
        status, raw, _ = self.call(base, "/api/batch/%s/report.html" % started["id"])
        self.assertIn(b"<img ", raw)

    def test_only_the_newest_batches_are_kept(self):
        base = self.start()
        with mock.patch.object(batch, "MAX_JOBS", 2):
            ids = []
            for i in range(3):
                _, started = self.submit(base, [("1.png", image(str(i)))])
                self.wait(base, started["id"])
                ids.append(started["id"])
            _, listing = self.call_json(base, "/api/batch")
        self.assertEqual([j["id"] for j in listing["batches"]], [ids[2], ids[1]])
        self.assertEqual(self.call(base, "/api/batch/" + ids[0])[0], 404)


class Reports(unittest.TestCase):
    def job(self, embed=False):
        job = batch.Job(server.Config(backend="ollama"), "ocr", "llava:7b", "", embed,
                        [("a<b>.png", PNG), ("2.png", PNG)], [("notes.txt", "not a PNG, JPEG or WebP image")])
        job.items[0].update(status="ok", text="Rechnung ```code``` <script>alert(1)</script> äöü €", seconds=1.5, chars=50)
        job.items[1].update(status="error", error="the model crashed")
        job.status = "done"
        if embed:
            job.thumbnails[0] = ("image/png", PNG)
        return job

    def test_markdown_keeps_the_text_verbatim_inside_a_long_enough_fence(self):
        text = report.markdown(self.job())
        self.assertIn("````\nRechnung ```code``` <script>alert(1)</script> äöü €\n````", text)
        self.assertIn("a\\<b\\>.png", text)
        self.assertIn("the model crashed", text)
        self.assertIn("notes.txt", text)
        self.assertIn("llava:7b", text)
        self.assertNotIn("data:image", text)
        self.assertRegex(text, r"\*\*Date:\*\* \d{4}-\d\d-\d\d \d\d:\d\d \S+", "the time carries its zone")

    def test_html_escapes_everything_and_runs_no_script(self):
        text = report.html(self.job())
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", text)
        self.assertIn("a&lt;b&gt;.png", text)
        self.assertNotIn("<script", text)
        self.assertNotIn("<img", text)
        self.assertIn('<meta charset="utf-8">', text)

    def test_images_go_in_only_when_asked(self):
        self.assertIn('<img src="data:image/png;base64,', report.html(self.job(embed=True)))
        self.assertIn("](data:image/png;base64,", report.markdown(self.job(embed=True)))

    def test_pdf(self):
        import importlib.util
        if importlib.util.find_spec("fpdf") is None:   # a bare Python: the image has it, see the container run
            with self.assertRaises(backends.ApiError) as caught:
                report.pdf(self.job())
            self.assertEqual(caught.exception.status, 501)
            return
        data = report.pdf(self.job())
        self.assertTrue(data.startswith(b"%PDF-"))
        self.assertGreater(len(data), 1000)


if __name__ == "__main__":
    unittest.main()
