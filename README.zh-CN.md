# fermi-infer

**一个面向本地小语言模型的 Rust 原生推理引擎。**

`fermi-infer` 目前优先面向 Apple Silicon 本地推理场景，基于 Candle + Metal，提供 CLI、OpenAI 兼容 HTTP、gRPC 三个入口，并补齐了 benchmark、eval、metrics 等工程化基础能力。

## 项目现在能做什么

- 在 `Metal` / `CUDA` / `CPU` 上运行 decoder-only 模型
- 从 `config.json` 自动识别模型架构
- 首次运行时自动从 Hugging Face 下载模型
- 提供：
  - 终端交互式聊天
  - OpenAI 兼容 HTTP 服务
  - gRPC 流式服务
- 内置：
  - 基准测试工具 `fermi-bench`
  - 效果回归工具 `fermi-eval`
  - Prometheus 风格指标输出

## 当前定位

它已经不是 demo，但也还不是完整工业级推理引擎。

当前做得比较扎实的部分：

- 单请求本地推理
- 流式输出
- Qwen / Phi-3 / Llama 架构装载
- 按模型族区分 prompt template 和 stop token
- benchmark / eval 回归
- OpenAI / gRPC 服务化与基础可观测性

仍在继续补强的部分：

- batching / 调度
- paged KV cache
- 更强的高并发吞吐
- 更多模型族
- 更完整的生产治理

## 当前支持的模型架构

- `Qwen`
  - 支持 Qwen2.5 / Qwen3 风格 causal LM
  - 推荐起点：`Qwen/Qwen3-1.7B`
- `Llama`
  - 支持标准 safetensors 布局的 Llama-family 模型
  - 如果要做对话，建议直接使用 `meta-llama/Llama-3.2-1B-Instruct`
- `Phi-3`
  - 支持 Phi-3 / Phi-3.5 风格 causal LM

注意：

- 如果你要做对话或指令跟随，优先使用 `Instruct` 模型。
- `base` 模型虽然可能能加载，但很容易出现 prompt 回显、指令跟随差、对话质量不稳定等问题。

## 快速开始

### CLI 交互聊天

```bash
cargo run -p fermi-infer --release --features metal -- chat
```

### CLI 单轮问答

```bash
cargo run -p fermi-infer --release --features metal -- \
  --model Qwen/Qwen3-1.7B \
  --preset chat-precise \
  --prompt "请解释什么是第一性原理。"
```

### 启动 OpenAI 兼容服务

```bash
cargo run -p fermi-openai --release --features metal
```

然后本地调用：

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

### 跑 benchmark

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

### 跑 eval

```bash
cargo run -p fermi-eval --release --features metal -- \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --preset chat-precise \
  --json \
  --out reports/eval_llama_instruct.json
```

## Prompt 模板与采样预设

项目现在会按模型族自动选择 prompt 模板：

- `Qwen` / `Phi-3`：ChatML 风格
- `Llama`：Llama 3 风格 header / EOT token

支持的 sampling preset：

- `chat-balanced`
- `chat-precise`
- `reasoning`
- `creative`

示例：

```bash
cargo run -p fermi-infer --release --features metal -- \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --preset chat-precise \
  --prompt "Introduce Rust in exactly two sentences and mention memory safety."
```

## Hugging Face 模型下载

模型查找顺序：

1. `FERMI_MODEL_DIR`
2. `--model` 指向的本地目录
3. 本机 Hugging Face cache
4. 从 Hugging Face 在线下载

模型目录需要这些文件：

- `tokenizer.json`
- `config.json`
- `model.safetensors`
  - 或 `model.safetensors.index.json` + shard 文件

如果模型是 gated/private，需要先配置 token：

```bash
export HF_TOKEN=hf_your_token_here
# 或
export HUGGINGFACE_HUB_TOKEN=hf_your_token_here
```

然后先去模型页申请权限，例如：

- [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

## 配置

配置发现顺序：

- CLI `--config PATH`
- `FERMI_CONFIG=/path/to/fermi.toml`
- 当前目录 `./fermi.toml`

参数优先级：

- 请求 / CLI 参数
- 环境变量
- `fermi.toml`
- 内置默认值

建议从 `fermi.toml.example` 开始。

示例：

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

常用环境变量：

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

## OpenAI 兼容 API

当前提供：

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/models`
- `GET /metrics`

`thinking` 参数支持：

- `on`
- `off`
- `auto`

实际行为会受这些因素共同影响：

- 请求里的 `thinking`
- 模型本身是否支持
- `FERMI_SUPPORTS_THINKING`
- `FERMI_DEFAULT_THINKING`
- `FERMI_DISABLE_THINK`

## Metrics

OpenAI 服务会在同一个 HTTP 地址上暴露：

- `GET /metrics`

gRPC 可通过单独地址暴露 metrics：

```bash
export FERMI_METRICS_ADDR=0.0.0.0:9100
```

当前已经覆盖的指标包括：

- 请求总数
- 错误数
- 活跃请求数
- 排队等待时间
- TTFT
- 生成 token 总数
- 生成耗时
- 平均 tokens/s

## Workspace 结构

- `crates/fermi-runtime`：推理引擎、配置、session、prompting、sampling
- `crates/fermi-models`：模型实现
- `crates/fermi-io`：模型发现、加载、Hugging Face 下载
- `crates/fermi-cli`：终端交互入口
- `crates/fermi-openai`：OpenAI 兼容 HTTP 服务
- `crates/fermi-grpc`：gRPC 流式服务
- `crates/fermi-bench`：benchmark 工具
- `crates/fermi-eval`：效果回归工具
- `crates/fermi-metrics`：指标采集

## 文档

- 英文 README：[README.md](README.md)
- 项目总览：[docs/fermi-project-overview.zh-CN.md](docs/fermi-project-overview.zh-CN.md)
- 分享稿：[docs/fermi-infer-tech-blog-share.zh-CN.md](docs/fermi-infer-tech-blog-share.zh-CN.md)
- Qwen 深挖：[docs/qwen-architecture-tech-blog.zh-CN.md](docs/qwen-architecture-tech-blog.zh-CN.md)

## 开发

```bash
cargo check
cargo test
```

更聚焦的验证方式：

```bash
cargo test -p fermi-runtime -p fermi-bench -p fermi-eval -p fermi-openai -p fermi-grpc
```

## License

Apache-2.0
