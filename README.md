# Fermi Infer

**A Rust-native local inference engine for small language models.**

`fermi-infer` targets fast local inference on Apple Silicon first, with Candle-based Metal acceleration, a modular Rust workspace, and service entrypoints for CLI, OpenAI-compatible HTTP, and gRPC.

## What It Does

- Runs decoder-only LLMs locally with `Metal`, `CUDA`, or `CPU`
- Auto-detects model architecture from `config.json`
- Downloads Hugging Face models automatically on first run
- Exposes:
  - interactive CLI chat
  - OpenAI-compatible HTTP API
  - gRPC streaming API
- Includes built-in:
  - benchmark tool: `fermi-bench`
  - evaluation harness: `fermi-eval`
  - Prometheus-style metrics

## Current Status

The project is beyond prototype stage, but it is not yet a full industrial inference engine in the `vLLM / TensorRT-LLM` sense.

What is solid today:

- single-request local inference
- streaming generation
- Qwen / Phi-3 / Llama-family architecture loading
- architecture-specific prompt templates and stop tokens
- benchmark and regression-eval tooling
- OpenAI/gRPC serving with basic observability

What is still in progress:

- advanced batching / scheduling
- paged KV cache
- stronger multi-request throughput
- broader model-family coverage
- deeper production governance

## Supported Architectures

- `Qwen`:
  - Qwen2.5 / Qwen3 style causal LM checkpoints
  - recommended starting point: `Qwen/Qwen3-1.7B`
- `Llama`:
  - Llama-family checkpoints with standard safetensors layout
  - for chat, use an `Instruct` checkpoint such as `meta-llama/Llama-3.2-1B-Instruct`
- `Phi-3`:
  - Phi-3 / Phi-3.5 style causal LM checkpoints

Important:

- For chat and instruction-following, prefer `Instruct` models.
- Base checkpoints can load, but they often echo prompts or behave poorly in chat mode.

## Quick Start

### Interactive CLI chat

```bash
cargo run -p fermi-infer --release --features metal -- chat
```

### Single prompt

```bash
cargo run -p fermi-infer --release --features metal -- \
  --model Qwen/Qwen3-1.7B \
  --preset chat-precise \
  --prompt "Explain what first-principles thinking is."
```

### OpenAI-compatible server

```bash
cargo run -p fermi-openai --release --features metal
```

Then call it locally:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-1.7B",
    "messages": [
      {"role": "system", "content": "You are a precise technical assistant."},
      {"role": "user", "content": "Explain Rust ownership in plain English."}
    ],
    "stream": false,
    "temperature": 0.2,
    "max_tokens": 256
  }'
```

### Benchmark

```bash
cargo run -p fermi-bench --release --features metal -- \
  --model Qwen/Qwen3-1.7B \
  --suite all \
  --preset chat-precise \
  --runs 3 \
  --warmup 1 \
  --json \
  --out reports/bench_suite.json
```

### Evaluation

```bash
cargo run -p fermi-eval --release --features metal -- \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --preset chat-precise \
  --json \
  --out reports/eval_llama_instruct.json
```

## Prompting And Presets

`fermi-infer` now applies prompt formatting by model family:

- `Qwen` / `Phi-3`: ChatML-style formatting
- `Llama`: Llama 3 style headers / EOT tokens

Available sampling presets:

- `chat-balanced`
- `chat-precise`
- `reasoning`
- `creative`

Example:

```bash
cargo run -p fermi-infer --release --features metal -- \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --preset chat-precise \
  --prompt "Introduce Rust in exactly two sentences and mention memory safety."
```

## Hugging Face Model Download

Model discovery order:

1. `FERMI_MODEL_DIR`
2. local path passed as `--model`
3. local Hugging Face cache
4. Hugging Face download

Required files:

- `tokenizer.json`
- `config.json`
- `model.safetensors`
  - or `model.safetensors.index.json` plus shard files

If a model is gated or private, configure a token first:

```bash
export HF_TOKEN=hf_your_token_here
# or
export HUGGINGFACE_HUB_TOKEN=hf_your_token_here
```

Then request access on the model page, for example:

- [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

## Configuration

Config discovery order:

- CLI `--config PATH`
- `FERMI_CONFIG=/path/to/fermi.toml`
- `./fermi.toml`

Parameter precedence:

- request / CLI argument
- environment variable
- `fermi.toml`
- built-in default

Start from `fermi.toml.example`.

Example:

```toml
[model]
id = "Qwen/Qwen3-1.7B"
offline = false

[generation]
default_max_new_tokens = 256
max_new_tokens_cap = 9056
temperature = 0.3
top_p = 0.95
repeat_penalty = 1.1

[cli]
timeout_ms = 60000
default_system_prompt = ""
default_system_prompt_file = "prompts/system.txt"
disable_think = false

[openai]
addr = "0.0.0.0:8000"
engine_pool = 1
default_system_prompt = ""
default_system_prompt_file = "prompts/system.txt"
default_thinking = "off"
supports_thinking = true
disable_think = false

[grpc]
addr = "0.0.0.0:50051"
engine_pool = 1
timeout_ms = 60000
session_ttl_ms = 0
session_max = 0
default_system_prompt = ""
default_system_prompt_file = "prompts/system.txt"
disable_think = false
```

Useful environment overrides:

- `FERMI_MODEL`
- `FERMI_OFFLINE`
- `HF_HUB_OFFLINE`
- `FERMI_ENGINE_POOL`
- `FERMI_OPENAI_ADDR`
- `FERMI_GRPC_ADDR`
- `FERMI_TIMEOUT_MS`
- `FERMI_SESSION_TTL_MS`
- `FERMI_SESSION_MAX`
- `FERMI_DEFAULT_SYSTEM_PROMPT`
- `FERMI_DEFAULT_SYSTEM_PROMPT_FILE`
- `FERMI_DEFAULT_THINKING`
- `FERMI_SUPPORTS_THINKING`
- `FERMI_DISABLE_THINK`
- `FERMI_DEFAULT_MAX_NEW_TOKENS`
- `FERMI_MAX_NEW_TOKENS_CAP`
- `FERMI_DEFAULT_TEMPERATURE`
- `FERMI_DEFAULT_TOP_P`
- `FERMI_DEFAULT_REPEAT_PENALTY`
- `FERMI_METRICS_ADDR`

## OpenAI-Compatible API

Available endpoints:

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/models`
- `GET /metrics`

`thinking` is supported on the OpenAI-compatible chat API:

- `on`
- `off`
- `auto`

The final behavior depends on:

- request value
- model capability
- `FERMI_SUPPORTS_THINKING`
- `FERMI_DEFAULT_THINKING`
- `FERMI_DISABLE_THINK`

## Metrics

OpenAI server exposes metrics on the same HTTP address:

- `GET /metrics`

gRPC can expose a dedicated metrics server with:

```bash
export FERMI_METRICS_ADDR=0.0.0.0:9100
```

Tracked metrics currently include:

- request count
- request error count
- active requests
- queue wait time
- TTFT
- generated token count
- generation duration
- average tokens per second

## Workspace Layout

- `crates/fermi-runtime`: inference engine, config, sessions, prompting, sampling
- `crates/fermi-models`: model implementations
- `crates/fermi-io`: model discovery, loading, Hugging Face download
- `crates/fermi-cli`: terminal chat interface
- `crates/fermi-openai`: OpenAI-compatible HTTP server
- `crates/fermi-grpc`: gRPC streaming server
- `crates/fermi-bench`: benchmark tool
- `crates/fermi-eval`: evaluation harness
- `crates/fermi-metrics`: metrics collection

## Docs

- Chinese README: [README.zh-CN.md](README.zh-CN.md)
- Project overview: [docs/fermi-project-overview.zh-CN.md](docs/fermi-project-overview.zh-CN.md)
- Technical blog: [docs/fermi-infer-tech-blog-share.zh-CN.md](docs/fermi-infer-tech-blog-share.zh-CN.md)
- Qwen deep dive: [docs/qwen-architecture-tech-blog.zh-CN.md](docs/qwen-architecture-tech-blog.zh-CN.md)

## Development

```bash
cargo check
cargo test
```

For focused validation:

```bash
cargo test -p fermi-runtime -p fermi-bench -p fermi-eval -p fermi-openai -p fermi-grpc
```

## License

Apache-2.0
