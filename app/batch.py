"""Batch mode: many images, or ZIP archives of images, through one model, one image after the other.

Nothing is written to disk: an image is kept in memory until its turn and dropped right after it
(only a small preview stays when the report should show the images), and finished batches stay in
memory until the server restarts. Reports are made on request, see app.report.
"""
import email.parser
import email.policy
import io
import re
import threading
import time
import uuid
import zipfile
import zlib

from app import backends
from app import server as host  # circular import: only used at call time

MAX_FILES = 500          # images in one batch
MAX_JOBS = 20            # batches kept in memory; the oldest finished ones are dropped
PREVIEW_PIXELS = 1024    # longest side of an image in a report
JOB_ID = re.compile(r"^[0-9a-f]{12}$")


def _natural(name):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def parse_upload(content_type, body):
    """multipart/form-data -> [(file name, bytes)]. Only the base name is kept; it is never used as a path."""
    if not content_type.lower().startswith("multipart/form-data"):
        raise backends.ApiError(415, "Send the files as multipart/form-data")
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1", "replace") + b"\r\n\r\n" + body)
    files = []
    if message.is_multipart():
        for part in message.iter_parts():
            filename = part.get_filename()
            if filename:
                files.append((filename.replace("\\", "/").rsplit("/", 1)[-1], part.get_payload(decode=True) or b""))
    if not files:
        raise backends.ApiError(400, "No files in the upload")
    return files


def collect(files, max_image_bytes, max_total_bytes):
    """Images out of uploaded files and ZIP archives -> (images [(name, bytes)], skipped [(name, reason)])."""
    images, skipped = [], []
    total = 0
    limit_text = "larger than the %.0f MB image limit" % (max_image_bytes / 1024 / 1024)

    def add(name, data):
        nonlocal total
        if host.sniff_image_type(data) is None:
            skipped.append((name, "not a PNG, JPEG or WebP image"))
        elif len(data) > max_image_bytes:
            skipped.append((name, limit_text))
        elif total + len(data) > max_total_bytes:
            skipped.append((name, "the batch is full"))
        else:
            total += len(data)
            images.append((name, data))

    for name, data in files:
        if not zipfile.is_zipfile(io.BytesIO(data)):
            add(name, data)
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for info in archive.infolist():
                    inner = info.filename
                    base = inner.rstrip("/").rsplit("/", 1)[-1]
                    if info.is_dir() or inner.startswith("__MACOSX/") or base.startswith("."):
                        continue
                    label = name + "/" + inner
                    if base.lower().endswith(".zip"):
                        skipped.append((label, "archives inside archives are not opened"))
                    elif info.file_size > max_image_bytes:
                        skipped.append((label, limit_text))
                    elif total + info.file_size > max_total_bytes:
                        skipped.append((label, "the batch is full"))
                    else:
                        try:
                            with archive.open(info) as handle:
                                content = handle.read(max_image_bytes + 1)   # the size in the header may lie
                        except (RuntimeError, zipfile.BadZipFile, zlib.error, OSError, EOFError) as error:
                            skipped.append((label, "cannot be read: %s" % type(error).__name__))
                            continue
                        add(label, content)
        except (zipfile.BadZipFile, zlib.error, OSError, EOFError):
            skipped.append((name, "damaged ZIP archive"))
    images.sort(key=lambda entry: _natural(entry[0]))
    if len(images) > MAX_FILES:
        skipped.extend((name, "more than %d images" % MAX_FILES) for name, _ in images[MAX_FILES:])
        images = images[:MAX_FILES]
    return images, skipped


def preview(data):
    """A small JPEG for the report, or the image itself when Pillow is not installed; None if it cannot be read."""
    try:
        from PIL import Image
    except ImportError:
        return host.sniff_image_type(data), data
    try:
        with Image.open(io.BytesIO(data)) as picture:
            picture.thumbnail((PREVIEW_PIXELS, PREVIEW_PIXELS))
            out = io.BytesIO()
            picture.convert("RGB").save(out, "JPEG", quality=80)
            return "image/jpeg", out.getvalue()
    except Exception:  # noqa: BLE001 - a broken image only loses its preview
        return None


class Job:
    def __init__(self, config, mode, model, prompt, embed, images, skipped):
        self.id = uuid.uuid4().hex[:12]
        self.config, self.mode, self.model, self.prompt, self.embed = config, mode, model, prompt, embed
        self.images = [data for _, data in images]          # each one becomes None after its turn
        self.skipped = list(skipped)
        self.thumbnails = {}                                 # item index -> (mime, bytes), only when embed
        self.items = [{"name": name, "status": "pending", "text": "", "error": None, "seconds": None, "chars": 0}
                      for name, _ in images]
        self.status = "running"
        self.created_at = time.time()
        self._started, self._finished = time.monotonic(), None
        self.cancel_requested = threading.Event()
        self.lock = threading.Lock()

    def execute(self, runs):
        try:
            for index, data in enumerate(self.images):
                if self.cancel_requested.is_set():
                    break
                self._run_one(index, data)
        finally:
            with self.lock:
                for index, item in enumerate(self.items):
                    self.images[index] = None
                    if item["status"] in ("pending", "running"):
                        item["status"] = "cancelled"
                self.status = "cancelled" if self.cancel_requested.is_set() else "done"
                self._finished = time.monotonic()
            runs.end_analysis()

    def _run_one(self, index, data):
        item = self.items[index]
        with self.lock:
            item["status"] = "running"
        started = time.monotonic()
        try:
            answer = host.analyze(self.config, data, self.mode, self.model, self.prompt)
            with self.lock:
                item.update(status="ok", text=answer["text"], chars=len(answer["text"]))
        except backends.ApiError as error:
            with self.lock:
                item.update(status="error", error=error.message)
        except Exception as error:  # noqa: BLE001 - one image must not end the batch
            with self.lock:
                item.update(status="error", error="Internal error: %s: %s" % (type(error).__name__, error))
        finally:
            with self.lock:
                item["seconds"] = round(time.monotonic() - started, 1)
                if self.embed:
                    picture = preview(data)
                    if picture:
                        self.thumbnails[index] = picture
                self.images[index] = None

    def cancel(self):
        self.cancel_requested.set()

    def snapshot(self):
        with self.lock:
            end = self._finished or time.monotonic()
            return {
                "id": self.id, "status": self.status, "mode": self.mode, "model": self.model,
                "prompt": self.prompt or None, "embed_images": self.embed, "created_at": self.created_at,
                "elapsed_seconds": round(end - self._started, 1), "total": len(self.items),
                "done": sum(1 for i in self.items if i["status"] in ("ok", "error")),
                "items": [dict(i) for i in self.items],
                "skipped": [{"name": name, "reason": reason} for name, reason in self.skipped],
            }

    def summary(self):
        with self.lock:
            return {"id": self.id, "status": self.status, "mode": self.mode, "model": self.model,
                    "created_at": self.created_at, "total": len(self.items),
                    "ok": sum(1 for i in self.items if i["status"] == "ok"),
                    "errors": sum(1 for i in self.items if i["status"] == "error")}


class Batches:
    """The batches of this server, in memory only. One runs at a time."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = []   # oldest first

    def start(self, config, mode, model, prompt, embed, images, skipped, runs):
        with self.lock:
            if any(job.status == "running" for job in self.jobs):
                raise backends.ApiError(409, "A batch is already running; start the next one when it has finished")
            runs.begin_analysis()   # refused while a benchmark runs; a benchmark waits for the batch
            job = Job(config, mode, model, prompt, embed, images, skipped)
            self.jobs.append(job)
            finished = [j for j in self.jobs if j.status != "running"]
            for old in finished[:max(0, len(self.jobs) - MAX_JOBS)]:
                self.jobs.remove(old)
        threading.Thread(target=job.execute, args=(runs,), daemon=True).start()
        return job

    def get(self, job_id):
        if not JOB_ID.match(job_id):
            return None
        with self.lock:
            return next((job for job in self.jobs if job.id == job_id), None)

    def listing(self):
        with self.lock:
            jobs = list(self.jobs)
        return [job.summary() for job in reversed(jobs)]
