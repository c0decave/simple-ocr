# simple-ocr

Drop an image, get its text (OCR) or a description from a local vision model on LM Studio, Ollama,
vLLM or llama.cpp (`llama-server`).
Python standard library for the app, plus fpdf2 (pinned with hashes, in the image) for PDF reports. No Node.

## Run

```
cp .env.example .env      # then edit host, port and API key
./run.sh
```

`run.sh` builds the image and starts it with `podman`, then open <http://127.0.0.1:8080>.
Drag and drop, click, or paste (Ctrl+V) a PNG, JPEG or WebP. OCR is the default mode.
Changing mode or model re-runs the current image, and the Send button repeats it after an error.
Under "Prompt" the prompt sent with the image can be changed (unless `CUSTOM_PROMPT=false`); a changed
prompt is kept when the mode changes, and "Back to the default prompt" restores it.

The model list shows each vision model with its size and whether it is loaded. Below the result,
a table lists everything loaded on the model server right now (type, parameters, quantization, size, total).
The selected model is highlighted, and if it is not loaded yet it is shown with the memory it adds.

With compose (`docker compose` or `podman compose`, it reads `.env` too):

```
docker compose up -d --build
```

Without the script:

```
podman build -t simple-ocr:$(cat VERSION) .
podman run --rm --env-file .env -p 127.0.0.1:8080:8080 -v simple-ocr-data:/data simple-ocr:$(cat VERSION)
```

Saved benchmark runs live in the named volume `simple-ocr-data` (mounted at `/data`). `run.sh` and
`compose.yaml` use the same volume, so runs survive restarts, `compose down` and `up --build`;
only `compose down -v` or `podman volume rm simple-ocr-data` delete them.

## Package for another machine

```
./dist.sh
```

builds `dist/simple-ocr-<version>.tar.gz` (plus a `.sha256`). Copy it over, then:

```
tar -xzf simple-ocr-<version>.tar.gz && cd simple-ocr-<version>
cp .env.example .env      # then edit it
./start.sh                # loads image.tar once, needs only podman or docker
```

It also writes `dist/simple-ocr-<version>-source.tar.gz`, the committed source without git history.
Both are built from the committed code only (`git archive HEAD`, it refuses uncommitted changes):
no `.env`, no saved runs and no images go in. `COMMIT` names the commit, and `compose.yaml` in the
package uses the loaded image instead of building one.

Before anything is built, `dist.sh` scans the release and stops if it finds private network
addresses (10.x, 172.16-31.x, 192.168.x), mail addresses, private keys, home paths, or a word from
`.private-words` (a local, git-ignored list, one word per line, e.g. your host names). Examples use
the documentation addresses 192.0.2.x and `.example` names instead.

On a remote server set `APP_BIND=0.0.0.0` in `.env` to reach the page from other machines.

Without a container: `PORT=8080 python3 -m app.server` reads `.env` from the current directory
(or the file named in `ENV_FILE`). Real environment variables win over the file.

## Batch page (`/batch`)

Many images at once: drop images and ZIP archives (folders inside are read, other files are listed as
skipped), pick the task, the model and optionally a custom prompt, and start. The images run one after
the other through the model; the page shows each result as it comes and can cancel after the current
image. The report downloads as Markdown, HTML or PDF, with the images as small previews when
"Embed the images in the report" is ticked.

Nothing is written to disk: each image is dropped from memory right after its turn (only the preview
stays when it was asked for), and finished batches stay in memory until the server restarts (the 20
newest). One batch runs at a time, and a batch and a benchmark exclude each other.
At most 500 images and `MAX_BATCH_MB` per upload.

API: `POST /api/batch?mode=ocr&model=...&prompt=...&images=1` with the files as multipart/form-data
(`curl -F files=@a.png -F files=@scans.zip ...`), then `GET /api/batch/<id>` for the state,
`POST /api/batch/<id>/cancel`, and `GET /api/batch/<id>/report.md|html|pdf`. `GET /api/batch` lists them.

The PDF uses the DejaVu fonts: Latin, Greek and Cyrillic text is complete, characters outside them
(for example Chinese or Japanese) are left out of the PDF; the HTML and Markdown reports keep them.

## Benchmark page (`/debug`)

Compare models on one image: pick an image, the task (OCR or describe) and any or all vision models.
For every model the run does the same steps in this order:

1. unload everything that is loaded on the model server and check that nothing is left,
2. load the model and time it,
3. run the image and time it,

and after the last model it unloads that one too. By default the models that were loaded before
the run are loaded again at the end (with the server's default settings, not their old load options).

vLLM, and llama.cpp started with one fixed model, cannot load or unload models: there the selected
models only run one after the other, without load times and without restore. llama.cpp in router
mode loads and unloads like LM Studio and Ollama.

The page shows a statistics table (size, load time, inference time, total, characters, group, status;
click a header to sort, default is load time) and a comparison: results are grouped by word-level
similarity (identical, similar from 80 % matching words, different), a matrix shows every pair,
and a side-by-side diff marks the changed words (it compares the first 3000 lines; longer lines are marked as a whole). Failed models are listed and do not stop the run.
Charts show where each model lies: a sideways similarity tree (average linkage on word-level
similarity, models that merge further left gave more similar text), speed against agreement
(mean similarity to all other models; the X axis can be inference, load or total time or size),
and load plus inference time per model.

A run can be cancelled after the current model. Every finished run is saved as
`<DATA_DIR>/runs/<id>.json` (the same JSON as the export button). The "Saved runs" list on the
page shows them, newest first, with buttons to show a run again (statistics, charts, diff) or to
download its JSON. After a restart the page opens the newest saved run.
API: `GET /api/debug/saved` lists them, `GET /api/debug/runs/<id>` returns one.

While a run is active the main page cannot send images (it would load models in between), and only
one run can be active. **Starting a run unloads every model on the model server**, also those
that other people or tools may be using.

## Model loading lock

`MODEL_LOADING_ALLOWED` in `app/backends.py` is hard-coded on purpose, not an `.env` setting. It is
`True` at the moment: models are loaded on first use, and the benchmark unloads and loads them.

Set it to `False` and rebuild to lock loading: the interface then never makes the model server load or
unload a model. Every vision model is listed, but only the loaded ones can be chosen, on the main page
and in the benchmark, and the API refuses the others (HTTP 403) before anything reaches the model
server. The benchmark then runs the loaded models one after the other, without unloading, loading or
restoring. A model that someone unloads on the server between the check and the request would still
be loaded by LM Studio (JIT loading) or Ollama; switch JIT loading off in LM Studio as a second safeguard.

## Model servers

| Server | Model list | Vision detection | Load / unload in the benchmark |
|---|---|---|---|
| LM Studio | `/api/v1/models` | `capabilities.vision` | yes |
| Ollama | `/api/tags`, `/api/ps` | `/api/show` capabilities | yes (`keep_alive: 0` unloads) |
| vLLM | `/v1/models` | not reported, every served model is offered | no |
| llama.cpp | `/models` or `/v1/models` | `architecture.input_modalities` | router mode only |

Uploaded images are kept in memory only, for the request or the benchmark run, and are never written
to disk; saved runs contain the texts and timings, not the image.

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `BACKEND` | `lmstudio` | `lmstudio`, `ollama`, `vllm` or `llamacpp` |
| `LMSTUDIO_HOST` | `localhost` | LM Studio host name or IP, may include `:port` |
| `LMSTUDIO_PORT` | `1234` | LM Studio port |
| `LMSTUDIO_API_KEY` | empty | sent as `Authorization: Bearer ...` when set |
| `LMSTUDIO_URL` | empty | full URL starting with `http://` or `https://`; wins over host and port |
| `OLLAMA_HOST`, `_PORT`, `_API_KEY`, `_URL`, `_TLS_VERIFY` | `localhost`, `11434` | the same for Ollama; `OLLAMA_HOST` may also be a URL |
| `VLLM_HOST`, `_PORT`, `_API_KEY`, `_URL`, `_TLS_VERIFY` | `localhost`, `8000` | the same for vLLM (`--api-key`) |
| `LLAMACPP_HOST`, `_PORT`, `_API_KEY`, `_URL`, `_TLS_VERIFY` | `localhost`, `8080` | the same for llama.cpp `llama-server` (`--api-key`) |
| `<SERVER>_TLS_VERIFY` | `true` | `false` accepts a self-signed https certificate (`LMSTUDIO_`, `OLLAMA_`, `VLLM_`, `LLAMACPP_`); the connection stays encrypted but the server is not authenticated, and the start log says so |
| `CUSTOM_PROMPT` | `true` | the prompt can be edited on the main page (sent as `prompt`, at most 4000 characters, empty = default); `false` removes the editor and the server answers a custom prompt with 403 |
| `BATCH_PAGE` | `true` | the Batch button and the batch page; `false` removes the button and the server answers `/batch` and every `/api/batch...` route with 403 |
| `MAX_BATCH_MB` | `200` | largest batch upload (images and ZIP archives together) |
| `DEBUG_PAGE` | `true` | the Debug button and the benchmark page; `false` removes the button and the server answers `/debug` and every `/api/debug/...` route with 403 |
| `DEFAULT_MODEL` | first loaded vision model | model preselected on the page |
| `DESCRIBE_LANGUAGE` | `German` | language of the "Describe image" answer |
| `MAX_UPLOAD_MB` | `20` | upload limit |
| `UPSTREAM_TIMEOUT` | `300` | seconds to wait for the model |
| `DATA_DIR` | empty (`/data` in the image) | finished benchmark runs are saved below it; empty = not saved |
| `APP_PORT` | `8080` | port on this machine (`run.sh`, `start.sh`, compose) |
| `APP_BIND` | `127.0.0.1` | address the port is published on; `0.0.0.0` makes it reachable from other machines, and the app has no login, so only on a trusted network. `run.sh` and `start.sh` put an IPv6 address in brackets; for compose write it as `[::1]` |

`.env` holds the API key: it is git-ignored, excluded from the image build, and the key is never
sent to the browser or written to logs.

## Tests

```
python3 -m unittest tests.test_server tests.test_benchmark tests.test_compare tests.test_backends tests.test_dist tests.test_batch
```

The tests use fakes of LM Studio, Ollama, vLLM and llama.cpp, built from their API docs, and need no network.
