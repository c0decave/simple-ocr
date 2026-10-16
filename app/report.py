"""Batch reports as Markdown, HTML or PDF. The texts are shown as the model returned them."""
import base64
import datetime
import html as html_lib
import io
import os
import re

from app import backends

FONT_DIR = "/usr/share/fonts/truetype/dejavu"   # Debian package fonts-dejavu-core, installed in the image
TASKS = {"ocr": "Extract text (OCR)", "describe": "Describe image"}
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+!|])")


def _facts(job):
    state = job.snapshot()
    ok = sum(1 for i in state["items"] if i["status"] == "ok")
    errors = sum(1 for i in state["items"] if i["status"] == "error")
    return state, [
        ("Date", datetime.datetime.fromtimestamp(state["created_at"]).astimezone().strftime("%Y-%m-%d %H:%M %Z")),
        ("Model", state["model"]),
        ("Task", TASKS.get(state["mode"], state["mode"])),
        ("Prompt", "custom" if state["prompt"] else "default"),
        ("Images", "%d, %d ok, %d failed, %d skipped%s" % (
            len(state["items"]), ok, errors, len(state["skipped"]),
            "" if state["status"] == "done" else " (batch %s)" % state["status"])),
    ]


def _status_line(item):
    parts = [item["status"]]
    if item["seconds"] is not None:
        parts.append("%.1f s" % item["seconds"])
    if item["status"] == "ok":
        parts.append("%d characters" % item["chars"])
    return " · ".join(parts)


def _data_url(mime, data):
    return "data:%s;base64,%s" % (mime, base64.b64encode(data).decode())


def markdown(job):
    state, facts = _facts(job)
    escape = lambda text: _MARKDOWN_SPECIAL.sub(r"\\\1", text)  # noqa: E731
    out = ["# simple-ocr batch report", ""]
    out += ["- **%s:** %s" % (key, escape(value)) for key, value in facts]
    if state["prompt"]:
        out += ["", "Prompt:", "", _fence(state["prompt"])]
    for number, item in enumerate(state["items"], 1):
        out += ["", "## %d. %s" % (number, escape(item["name"])), "", "_%s_" % _status_line(item), ""]
        picture = job.thumbnails.get(number - 1)
        if picture:
            out += ["![%s](%s)" % (escape(item["name"]), _data_url(*picture)), ""]
        out.append(_fence(item["text"]) if item["status"] == "ok" else "> " + escape(item["error"] or item["status"]))
    if state["skipped"]:
        out += ["", "## Skipped files", ""]
        out += ["- %s: %s" % (escape(s["name"]), s["reason"]) for s in state["skipped"]]
    return "\n".join(out) + "\n"


def _fence(text):
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return "%s\n%s\n%s" % (fence, text, fence)


def html(job):
    state, facts = _facts(job)
    esc = lambda text: html_lib.escape(str(text), quote=True)  # noqa: E731
    out = ['<!doctype html>', '<html lang="en">', '<head>', '<meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width, initial-scale=1">',
           '<title>simple-ocr batch report</title>',
           '<style>',
           'body { font:15px/1.5 system-ui, sans-serif; color:#1c2128; background:#fff; max-width:960px; margin:0 auto; padding:24px 16px; }',
           'h1 { font-size:1.4rem; } h2 { font-size:1.1rem; margin:28px 0 4px; overflow-wrap:anywhere; }',
           'dl { display:grid; grid-template-columns:max-content 1fr; gap:2px 16px; } dt { color:#667085; } dd { margin:0; }',
           '.status { color:#667085; font-size:.875rem; } .error { color:#c0392b; }',
           'pre { white-space:pre-wrap; overflow-wrap:anywhere; background:#f6f7f9; border:1px solid #d0d5dd; border-radius:6px; padding:10px; font:13px/1.5 ui-monospace, monospace; }',
           'img { max-width:100%; max-height:480px; border:1px solid #d0d5dd; border-radius:6px; display:block; margin:8px 0; }',
           'section { break-inside:avoid-page; }',
           '</style>', '</head>', '<body>', '<h1>simple-ocr batch report</h1>', '<dl>']
    out += ["<dt>%s</dt><dd>%s</dd>" % (esc(key), esc(value)) for key, value in facts]
    out.append("</dl>")
    if state["prompt"]:
        out.append("<p>Prompt:</p><pre>%s</pre>" % esc(state["prompt"]))
    for number, item in enumerate(state["items"], 1):
        out.append("<section><h2>%d. %s</h2><div class=\"status\">%s</div>" % (number, esc(item["name"]), esc(_status_line(item))))
        picture = job.thumbnails.get(number - 1)
        if picture:
            out.append('<img src="%s" alt="%s">' % (_data_url(*picture), esc(item["name"])))
        if item["status"] == "ok":
            out.append("<pre>%s</pre>" % esc(item["text"]))
        else:
            out.append('<p class="error">%s</p>' % esc(item["error"] or item["status"]))
        out.append("</section>")
    if state["skipped"]:
        out.append("<h2>Skipped files</h2><ul>")
        out += ["<li>%s: %s</li>" % (esc(s["name"]), esc(s["reason"])) for s in state["skipped"]]
        out.append("</ul>")
    out += ["</body>", "</html>"]
    return "\n".join(out) + "\n"


def pdf(job):
    try:
        from fpdf import FPDF
    except ImportError:
        raise backends.ApiError(501, "PDF reports need fpdf2, which the container image installs")
    regular, bold = os.path.join(FONT_DIR, "DejaVuSans.ttf"), os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")
    if not (os.path.exists(regular) and os.path.exists(bold)):
        raise backends.ApiError(501, "PDF reports need the DejaVu fonts (Debian package fonts-dejavu-core)")
    state, facts = _facts(job)
    document = FPDF(format="A4")
    document.set_auto_page_break(True, margin=15)
    document.add_font("DejaVu", "", regular)
    document.add_font("DejaVu", "B", bold)
    document.set_title("simple-ocr batch report")
    document.add_page()
    width = document.epw

    def write(text, size=10, style="", gap=5, color=(28, 33, 40)):
        document.set_font("DejaVu", style, size)
        document.set_text_color(*color)
        document.multi_cell(width, gap, text, new_x="LMARGIN", new_y="NEXT")

    write("simple-ocr batch report", size=16, style="B", gap=9)
    for key, value in facts:
        write("%s: %s" % (key, value))
    if state["prompt"]:
        document.ln(2)
        write("Prompt:", style="B")
        write(state["prompt"], size=9)
    for number, item in enumerate(state["items"], 1):
        document.ln(4)
        write("%d. %s" % (number, item["name"]), size=12, style="B", gap=6)
        write(_status_line(item), size=9, color=(102, 112, 133))
        picture = job.thumbnails.get(number - 1)
        if picture:
            try:
                document.image(io.BytesIO(picture[1]), w=min(width, 120))
            except Exception:  # noqa: BLE001 - the text is what matters
                write("(the image could not be embedded)", size=9, color=(102, 112, 133))
        if item["status"] == "ok":
            write(item["text"] or " ", size=10)
        else:
            write(item["error"] or item["status"], color=(192, 57, 43))
    if state["skipped"]:
        document.ln(4)
        write("Skipped files", size=12, style="B", gap=6)
        for skipped in state["skipped"]:
            write("%s: %s" % (skipped["name"], skipped["reason"]), size=9)
    return bytes(document.output())
