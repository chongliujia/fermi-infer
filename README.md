# fermi-infer

Rust-based LLM inference tool with a workspace layout, a CLI entrypoint, and a gRPC server.

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

## OpenAI-like HTTP API

Start the server:

```bash
cargo run -p fermi-openai --release --features metal
```

Default listen address: `0.0.0.0:8000`

Supported endpoints:
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/models`

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

## Gradio demo client (Python)

Start the gRPC server first:

```bash
cargo run -p fermi-grpc --release --features metal
```

Then, from repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r demo/requirements.txt
python -m grpc_tools.protoc -I crates/fermi-grpc/proto \
  --python_out demo --grpc_python_out demo \
  crates/fermi-grpc/proto/fermi.proto
python demo/gradio_app.py
```

## Session behavior (minimal)

Runtime includes a minimal in-memory session store. It records `session_id` usage
and is wired into the gRPC handler. TTL/LRU cleanup and paging are not implemented yet.

## Notes

- Model files are downloaded from HuggingFace on first run and cached locally.
- Metal uses F16 weights by default; CPU uses F32.
- The gRPC build uses `tonic-build`, which requires `protoc` to be installed.
