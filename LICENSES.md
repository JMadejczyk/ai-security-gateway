# Licenses

Third-party terms for what the gateway ships and what `docker compose` runs. Versions are the ones pinned in `uv.lock` and `docker-compose.yml` on 2026-10-03; regenerate the Python table from installed package metadata when the lock changes.

## Container images

| Image | Version | License | Note |
| --- | --- | --- | --- |
| `redis` | 7.2.16 | BSD-3-Clause | The last BSD line. Redis 7.4 is RSALv2/SSPLv1 and 8.x adds AGPLv3; do not bump past 7.2 without a license decision (Valkey is the BSD alternative). |
| `postgres` | 16.15 | PostgreSQL License | Permissive. |
| `ollama/ollama` | 0.35.1 | MIT | Runtime only. Model weights are licensed separately: `qwen3:8b` is Apache-2.0. |
| `python` (base of our images) | 3.12.15-slim | PSF-2.0 plus Debian package licenses | |
| `ghcr.io/astral-sh/uv` (build stage only) | see `Dockerfile` | MIT OR Apache-2.0 | Not in the runtime image. |
| `grafana/grafana:12.4.12` | 12.4.12 (OSS) | AGPL-3.0-only | The OSS build, not Enterprise. Image based on Alpine (package licenses vary). |
| `grafana/loki:3.7.8` | 3.7.8 | AGPL-3.0-only | Loki's server code is AGPLv3; some client libraries in its repository are Apache-2.0. |
| `grafana/alloy:v1.20.1` | 1.20.1 | Apache-2.0 | Replaces Promtail (EOL) as the log shipper. Runs with `--disable-reporting`. |
| `prom/prometheus:v3.15.0` | 3.15.0 | Apache-2.0 | Busybox-based image (GPL-2.0 busybox binary, unmodified). |

**Grafana and Loki are AGPLv3.** We run the upstream images unmodified as separate services and talk to them only over their network APIs (provisioning files, HTTP queries, Loki push), so the AGPL places no obligation on this repository's code. Obligations would arise only if we modified Grafana or Loki and offered the modified version to users over a network; then the modified source would have to be offered to them. The dashboards and provisioning files in `grafana/` are our own content, loaded as data. Phone-home features are off: Grafana analytics, update checks, news feed and plugin preinstall (`GF_*` in `docker-compose.yml`), Loki `analytics.reporting_enabled: false`, Alloy `--disable-reporting`.

## Python runtime dependencies (gateway image)

All permissive. MPL-2.0 (certifi, tqdm) is file-level copyleft: it applies only if those files are modified.

| License | Packages |
| --- | --- |
| MIT | annotated-doc, annotated-types, anyio, attrs, catalogue, charset-normalizer, cloudpathlib, confection, cymem, fastapi, filelock, h11, httptools, jsonschema, jsonschema-specifications, markdown-it-py, mdurl, murmurhash, onnxruntime, preshed, presidio-analyzer, pydantic, pydantic-core, pydantic-settings, pyjwt, pyyaml, redis (redis-py), referencing, rich, rpds-py, setuptools, smart-open, spacy, spacy-legacy, spacy-loggers, sqlglot, srsly, thinc, typer, typing-inspection, urllib3, wasabi, watchfiles, weasel |
| BSD-2/3-Clause | blis, click, fsspec, httpcore, httpx, idna, jinja2, markupsafe, protobuf, pygments, python-dotenv, starlette, tldextract, uvicorn, websockets, wrapt, colorama (Windows only) |
| Apache-2.0 | flatbuffers, hf-xet, huggingface-hub, opentelemetry-api, phonenumbers, requests, requests-file, tokenizers |
| Mixed permissive | numpy (BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0), packaging (Apache-2.0 OR BSD-2-Clause), prometheus-client (Apache-2.0 AND BSD-2-Clause), regex (Apache-2.0 AND CNRI-Python), uvloop (Apache-2.0 / MIT), shellingham (ISC), typing-extensions (PSF-2.0) |
| MPL-2.0 | certifi; tqdm (MPL-2.0 AND MIT) |

Notes:

- **Presidio** runs with its no-op NLP engine; no spaCy model (whose weights have their own licenses) is downloaded or shipped. The Polish recognizers (PESEL, NIP) are ours.
- **tldextract** uses its bundled public-suffix snapshot (MPL-2.0 data from publicsuffix.org); nothing is fetched at runtime.
- **LiteLLM is not a dependency.** If it is added for commercial-model pricing, import only its MIT code: its `enterprise/` directory is under a separate license.

## Model weights

| Model | Revision | License | Note |
| --- | --- | --- | --- |
| `protectai/deberta-v3-base-prompt-injection-v2` (`onnx/model.onnx`, `onnx/tokenizer.json`, `onnx/config.json`) | `90c9989b1a342275dd0d1a95aad283c04e075671` | Apache-2.0 | The injection classifier. Not in the image or the repository: `models-init` fetches exactly these files at this commit into a volume, and the gateway checks the SHA-256 of each against `gateway/injection/model_manifest.json` before loading. Only the ONNX export is used (no pickle, no remote code). `huggingface-hub` and `hf-xet` are installed as dependencies of `tokenizers`; the gateway never downloads anything at runtime. |

## Demo MCP servers (`demo/mcp_servers`)

`mcp` 2.3.0 (MIT), `httpx` (BSD-3-Clause), `pydantic` (MIT), `pyjwt` (MIT), `sqlglot` (MIT), `psycopg[binary]` 3.3.6 and `psycopg-pool` 3.3.3 (LGPL-3.0-only; used unmodified as libraries, so our code carries no LGPL obligation beyond letting users swap the library — satisfied by installing it from PyPI in the image).

## Content we wrote

`config/feeds/signatures.json` is written by us. It is inspired by public descriptions (OWASP Top 10 for LLM applications, published MCP tool-poisoning write-ups) but copies no text from them.
