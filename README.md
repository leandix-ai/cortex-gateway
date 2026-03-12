# Cortex Gateway

> OpenAI-compatible proxy for Anthropic Claude with automatic complexity-based model routing and context compression.

## Features

- **Auto-Routing** — Automatically routes simple tasks to Haiku and complex tasks to Sonnet using heuristic + LLM classification
- **Context Compression** — 6-pass pipeline: sliding window, dedup file reads, summarize tool chains, prompt caching, routing, language detection
- **OpenAI-Compatible** — Drop-in replacement for OpenAI API endpoints (`/v1/chat/completions`, `/v1/models`)
- **SSE Streaming** — Full streaming support with `<think>` tag parsing into `reasoning_content`
- **Multi-Language** — Automatic language detection (Vietnamese, Chinese, Japanese, Korean, Arabic, Thai, Russian)
- **Prompt Caching** — Leverages Anthropic's prompt caching to reduce token costs

## Installation

```bash
pip install cortex-gateway
```

## Quick Start

```bash
# Start with defaults (0.0.0.0:8000)
cortex-gateway

# Custom host/port
cortex-gateway --host 127.0.0.1 --port 9000

# Dev mode with hot reload
cortex-gateway --reload

# Or run as module
python -m cortex_gateway
```

## Client Configuration

### Continue (config.yaml)

```yaml
models:
  - name: Cortex
    provider: openai
    model: cortex-auto           # auto-route between Haiku & Sonnet
    apiBase: http://localhost:8000/v1
    apiKey: sk-ant-...           # your Anthropic API key
```

### Custom Model Override

```yaml
models:
  - name: Cortex Sonnet
    provider: openai
    model: claude-sonnet-4-5   # bypass auto-routing
    apiBase: http://localhost:8000/v1
    apiKey: sk-ant-...
```

### Header-Based Model Config

```
X-Cortex-Model-Simple: claude-haiku-4-5
X-Cortex-Model-Complex: claude-sonnet-4-5
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming & non-streaming) |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check with pipeline config |

## Pipeline Passes

1. **Sliding Window** — Trims old conversation history
2. **Deduplicate Reads** — Removes duplicate file read results
3. **Summarize Chains** — Compresses settled tool call chains
4. **Inject Cache** — Adds Anthropic cache_control markers
5. **Router** — Classifies complexity (heuristic → LLM fallback)
6. **Language Detection** — Detects user language for response localization

## License

MIT
