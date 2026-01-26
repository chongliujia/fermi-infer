# fermi-infer

Rust-based LLM inference stack focused on **fast startup** and **fast responses** on macOS (Metal), with CLI, gRPC, and OpenAI-compatible HTTP APIs.

## Documentation

- **English (default):** `README.md`
- **中文文档:** `README.zh-CN.md`

## Highlights

- **Mac-first (Metal)**: optimized for Apple Silicon, F16 by default.
- **Simple serving**: CLI for local chat, gRPC for streaming tokens, OpenAI-like HTTP for drop-in clients.
- **HuggingFace models**: auto-download + cache, offline mode supported.
- **Thinking modes**: choose `thinking=on|off|auto` per request (OpenAI-like API).

## Quickstart (macOS / Metal)

```bash
cargo run -p fermi-openai --release --features metal
```

Then test with curl (streaming):

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"你好"}],
    "stream": true,
    "max_tokens": 256,
    "temperature": 0.7,
    "thinking": "off"
  }'
```

## OpenAI-like HTTP API

Start the server:

```bash
cargo run -p fermi-openai --release --features metal
```

Default listen address: `0.0.0.0:8000`

Supported endpoints:
- `POST /v1/chat/completions` (streaming SSE supported)
- `POST /v1/responses`
- `GET /v1/models`

### Thinking control

`/v1/chat/completions` accepts:
- `thinking: "on" | "off" | "auto"`  
  - `on`: model should wrap reasoning in `<think>...</think>`  
  - `off`: suppress reasoning  
  - `auto`: enable only when model supports thinking

Model support is inferred (Qwen/DeepSeek/R1/QwQ) and can be overridden:
- `FERMI_SUPPORTS_THINKING=1|0`
- `FERMI_DEFAULT_THINKING=on|off|auto`
- `FERMI_DISABLE_THINK=1` (force off)

## Gradio demo (OpenAI API)

Start the OpenAI server first:

```bash
cargo run -p fermi-openai --release --features metal
```

Then run the demo:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r demo/requirements.txt
python demo/gradio_app.py
```

## Build & run (CLI)

```bash
cargo run -p fermi-infer --release --features metal
```

CUDA build:

```bash
cargo run -p fermi-infer --release --features cuda
```

Interactive CLI commands:
- `/help` show commands
- `/reset` clear conversation context
- `/exit` quit

CLI flags:
- `--max-new-tokens N` maximum number of generated tokens (default: 1024)
- `--repeat-penalty P` repetition penalty (default: 1.0)

## Workspace layout

- `crates/fermi-cli` - CLI entrypoint
- `crates/fermi-grpc` - gRPC server (streaming tokens)
- `crates/fermi-models` - model implementations (Qwen3)
- `crates/fermi-runtime` - inference engine (prefill/decode + minimal session)
- `crates/fermi-io` - HuggingFace download + tokenizer + config load
- `crates/fermi-core` - placeholder for core utilities
- `crates/fermi-metrics` - placeholder for metrics

## Build & run (CLI)

From repo root:

```bash
cargo run -p fermi-infer --release --features metal
```

CUDA build:

```bash
cargo run -p fermi-infer --release --features cuda
```

Interactive CLI commands:
- `/help` show commands
- `/reset` clear conversation context
- `/exit` quit

CLI flags:
- `--max-new-tokens N` maximum number of generated tokens (default: 1024)
- `--repeat-penalty P` repetition penalty (default: 1.0)

## Build & run (gRPC)

The gRPC service streams tokens for a prompt and accepts a `session_id`.

```bash
cargo run -p fermi-grpc --release --features metal
```

Default listen address: `0.0.0.0:50051`

## gRPC API

Proto definition:

```
crates/fermi-grpc/proto/fermi.proto
```

Generate request fields:
- `session_id` - optional; if empty, the server creates one
- `prompt` - raw user prompt
- `max_new_tokens` - generation cap
- `repeat_penalty` - repetition penalty

Streaming response fields:
- `token` - decoded token text
- `token_id` - token id
- `session_id` - session id used for generation

## Session behavior (minimal)

Runtime includes a minimal in-memory session store. It records `session_id` usage
and is wired into the gRPC handler. TTL/LRU cleanup and paging are not implemented yet.

## Notes

- Model files are downloaded from HuggingFace on first run and cached locally.
- Metal uses F16 weights by default; CPU uses F32.
- The gRPC build uses `tonic-build`, which requires `protoc` to be installed.

## Configuration (env)

- `FERMI_MODEL` model id (default: `Qwen/Qwen3-1.7B`)
- `FERMI_OFFLINE=1` or `HF_HUB_OFFLINE=1` to disable downloads
- `FERMI_ENGINE_POOL` number of model instances (HTTP server)
- `FERMI_OPENAI_ADDR` bind address for HTTP server (default: `0.0.0.0:8000`)
- `FERMI_DEFAULT_THINKING` / `FERMI_SUPPORTS_THINKING` / `FERMI_DISABLE_THINK` (see above)

## Model support

Currently supported:
- **Qwen3** (tested with `Qwen/Qwen3-1.7B` on Metal)

Planned:
- Additional Qwen/DeepSeek variants
- Quantized weights (for faster cold start / lower memory)

## Project positioning

fermi-infer is a **Rust-first, Mac-friendly** inference stack:
- **Fast startup** for local/dev use
- **Fast streaming** for interactive chat
- **Simple serving** with OpenAI-compatible HTTP and gRPC

## Roadmap

- KV cache and attention optimizations for long context
- Throughput/latency benchmark tooling
- More model backends (Qwen/DeepSeek variants)
- OpenAI-style reasoning separation (optional)
- Metrics, tracing, and production-grade observability

## Contributing

PRs are welcome. A lightweight flow:
1. Fork and create a feature branch
2. Make changes and keep commits focused
3. Run `cargo fmt` (and `cargo clippy` if available)
4. Open a PR with a short description and screenshots/logs if relevant
