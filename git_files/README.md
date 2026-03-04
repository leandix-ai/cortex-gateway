# Leandix Cortex

An OpenAI-compatible context router that intelligently routes requests between Claude Haiku and Sonnet based on task complexity — saving cost without sacrificing quality.

Built for the [Continue](https://continue.dev) VS Code plugin, but works with any OpenAI-compatible client.

---

## How it works

Every request passes through a 6-stage pipeline before hitting the Anthropic API:

| Pass | Name | Description |
|------|------|-------------|
| 1 | Sliding window | Trims conversation history beyond the last N turns |
| 2 | Deduplicate reads | Collapses repeated identical tool results |
| 3 | Summarize tool chains | Compresses settled tool-use/result pairs in early history |
| 4 | Cache injection | Adds `cache_control` to eligible system messages |
| 5 | Router | Routes to Haiku (simple) or Sonnet (complex) |
| 6 | Language detection | Detects user language and instructs the model to reply in kind |

The router uses a two-stage strategy:
1. **Heuristic scoring** — fast, zero-cost regex pattern matching
2. **LLM classification** — only called when the heuristic score is inconclusive

---

## Project structure

```
server.py                  # Uvicorn entry point (CLI args, banner)
leandix_cortex/
├── __init__.py            # Re-exports app for uvicorn
├── pipeline.py            # Pass 1–4 + language detection (pure functions)
├── router.py              # Pass 5: complexity classifier
└── leandix_cortex.py      # HTTP layer: format converter, SSE streaming, FastAPI routes
```

---

## Getting started

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/leandix-cortex.git
cd leandix-cortex

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY
```

### 3. Run

```bash
python server.py
```

Options:

```bash
python server.py --host 0.0.0.0 --port 8000   # default
python server.py --reload                       # dev mode with hot reload
python server.py --workers 4                    # production multi-worker
```

---

## Continue plugin setup

In your Continue `config.yaml`:

```yaml
models:
  - name: Cortex
    provider: openai
    model: cortex-auto
    apiBase: http://localhost:8000/v1
    apiKey: sk-ant-your-key-here
```

---

## Model routing

By default, Cortex auto-routes between:

| Role | Default model |
|------|--------------|
| Simple tasks | `claude-haiku-4-5` |
| Complex tasks | `claude-sonnet-4-5` |

### Override via headers

```http
X-Cortex-Model-Simple: claude-haiku-4-5
X-Cortex-Model-Complex: claude-sonnet-4-5
```

### Force a specific model

Set `model` in the request body to any model name other than `cortex-auto`:

```json
{ "model": "claude-sonnet-4-5", ... }
```

---

## API

### `POST /v1/chat/completions`

OpenAI-compatible. Accepts `stream: true/false`.

Response includes a `cortex` metadata field:

```json
{
  "cortex": {
    "original_chars": 12400,
    "final_chars": 8100,
    "saved_chars": 4300,
    "compression_pct": 34.7,
    "routing": {
      "complexity": "simple",
      "classify_method": "heuristic",
      "model": "claude-haiku-4-5"
    }
  }
}
```

Response headers:

```
X-Cortex-Model: claude-haiku-4-5
X-Cortex-Complexity: simple
X-Cortex-Compression: 34.7%
X-Cortex-Language: vi
```

### `GET /v1/models`

Lists available model IDs.

### `GET /health`

Returns pipeline config and status.

---

## Docker

### Quick start

```bash
docker compose up -d
```

### Build & run manually

```bash
docker build -t leandix-cortex .
docker run -p 8000:8000 leandix-cortex
```

### Pass a fallback API key via env

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

```bash
docker compose up -d
```

> The API key is normally forwarded per-request from the client via `Authorization: Bearer`.
> The env var is optional and only used as a fallback.

---

## License

MIT
