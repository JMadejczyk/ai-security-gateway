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

Grafana and Loki (stage 14–18) are AGPLv3. Running them unmodified as separate services carries no obligation for this code; modifying and offering them over a network would.

## Python runtime dependencies (gateway image)

All permissive. MPL-2.0 (certifi, tqdm) is file-level copyleft: it applies only if those files are modified.

| License | Packages |
| --- | --- |
| MIT | annotated-doc, annotated-types, anyio, attrs, catalogue, charset-normalizer, cloudpathlib, confection, cymem, fastapi, filelock, h11, httptools, jsonschema, jsonschema-specifications, markdown-it-py, mdurl, murmurhash, preshed, presidio-analyzer, pydantic, pydantic-core, pydantic-settings, pyjwt, pyyaml, redis (redis-py), referencing, rich, rpds-py, setuptools, smart-open, spacy, spacy-legacy, spacy-loggers, sqlglot, srsly, thinc, typer, typing-inspection, urllib3, wasabi, watchfiles, weasel |
| BSD-2/3-Clause | blis, click, httpcore, httpx, idna, jinja2, markupsafe, pygments, python-dotenv, starlette, tldextract, uvicorn, websockets, wrapt, colorama (Windows only) |
| Apache-2.0 | opentelemetry-api, phonenumbers, requests, requests-file |
| Mixed permissive | numpy (BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0), packaging (Apache-2.0 OR BSD-2-Clause), prometheus-client (Apache-2.0 AND BSD-2-Clause), regex (Apache-2.0 AND CNRI-Python), uvloop (Apache-2.0 / MIT), shellingham (ISC), typing-extensions (PSF-2.0) |
| MPL-2.0 | certifi; tqdm (MPL-2.0 AND MIT) |

Notes:

- **Presidio** runs with its no-op NLP engine; no spaCy model (whose weights have their own licenses) is downloaded or shipped. The Polish recognizers (PESEL, NIP) are ours.
- **tldextract** uses its bundled public-suffix snapshot (MPL-2.0 data from publicsuffix.org); nothing is fetched at runtime.
- **LiteLLM is not a dependency.** If it is added for commercial-model pricing, import only its MIT code: its `enterprise/` directory is under a separate license.

## Demo MCP servers (`demo/mcp_servers`)

`mcp` 2.3.0 (MIT), `httpx` (BSD-3-Clause), `pydantic` (MIT), `pyjwt` (MIT), `sqlglot` (MIT), `psycopg[binary]` 3.3.6 and `psycopg-pool` 3.3.3 (LGPL-3.0-only; used unmodified as libraries, so our code carries no LGPL obligation beyond letting users swap the library — satisfied by installing it from PyPI in the image).

## Content we wrote

`feeds/signatures.json` is written by us. It is inspired by public descriptions (OWASP Top 10 for LLM applications, published MCP tool-poisoning write-ups) but copies no text from them.
