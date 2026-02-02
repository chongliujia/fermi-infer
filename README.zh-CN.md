# fermi-infer

**专为小语言模型 (SLMs) 打造的 Rust 原生推理引擎。**

在 Apple Silicon 上通过 Metal (GPU) 全加速，以极快的速度和零延迟启动运行高效模型（如 Qwen, SmolLM, Phi）。

## 文档

- English (default): `README.md`
- 中文文档：`README.zh-CN.md`

## 亮点

- **Mac 优先（Metal）**：针对 Apple Silicon 优化，默认 F16。
- **易用服务**：本地 CLI、gRPC 流式、OpenAI 兼容 HTTP。
- **HuggingFace 模型**：自动下载 + 本地缓存，支持离线。
- **thinking 模式**：`thinking=on|off|auto`（OpenAI API）。

## 快速开始（macOS / Metal）

```bash
cargo run -p fermi-openai --release --features metal
```

然后用 curl 测试（流式）：

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

## OpenAI 兼容 HTTP API

```bash
cargo run -p fermi-openai --release --features metal
```

默认地址：`0.0.0.0:8000`

支持端点：
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/models`

### thinking 控制

`/v1/chat/completions` 支持：
- `thinking: "on" | "off" | "auto"`
  - `on`：要求把思考放进 `<think>...</think>`
  - `off`：禁止思考输出
  - `auto`：模型支持则开启，否则关闭

可通过环境变量覆盖：
- `FERMI_SUPPORTS_THINKING=1|0`
- `FERMI_DEFAULT_THINKING=on|off|auto`
- `FERMI_DISABLE_THINK=1`（强制关闭）

## Gradio Demo（OpenAI API）

```bash
cargo run -p fermi-openai --release --features metal
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r demo/requirements.txt
python demo/gradio_app.py
```

## CLI 运行

```bash
cargo run -p fermi-infer --release --features metal
```

CUDA：

```bash
cargo run -p fermi-infer --release --features cuda
```

交互命令：
- `/help` 帮助
- `/reset` 清空上下文
- `/exit` 退出

CLI 参数：
- `--config PATH` 指定配置文件（默认自动发现 `./fermi.toml`）
- `--offline` / `--online` 强制离线/在线模型加载
- `--max-new-tokens N` 生成上限
- `--repeat-penalty P` 重复惩罚

## gRPC API

Proto 定义：

```
crates/fermi-grpc/proto/fermi.proto
```

## 会话（简化版）

当前是内存型 session，已支持 TTL / LRU 回收。

## 环境变量

推荐优先使用 `fermi.toml` 统一配置，再用环境变量做覆盖。

自动发现顺序：
- CLI `--config PATH`（仅 CLI）
- `FERMI_CONFIG=/path/to/fermi.toml`
- 当前目录 `./fermi.toml`

示例 `fermi.toml`：

```toml
[model]
id = "Qwen/Qwen3-1.7B"
offline = false

[generation]
default_max_new_tokens = 256
max_new_tokens_cap = 9056
temperature = 0.2
top_p = 1.0
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

可直接复制 `fermi.toml.example` 作为起点。

- `FERMI_MODEL` 模型 ID（默认：`Qwen/Qwen3-1.7B`）
- `FERMI_OFFLINE=1` / `HF_HUB_OFFLINE=1` 关闭在线下载
- `FERMI_ENGINE_POOL` HTTP 服务器实例数量
- `FERMI_OPENAI_ADDR` HTTP 监听地址
- `FERMI_GRPC_ADDR` gRPC 监听地址
- `FERMI_DEFAULT_SYSTEM_PROMPT` / `FERMI_DEFAULT_SYSTEM_PROMPT_FILE` 默认系统提示词（文本或文件）
- `FERMI_DEFAULT_THINKING` / `FERMI_SUPPORTS_THINKING` / `FERMI_DISABLE_THINK`
- `FERMI_SESSION_TTL_MS` gRPC 会话空闲 TTL（毫秒，未设置或 `0` 表示关闭）
- `FERMI_SESSION_MAX` gRPC 内存会话上限（超过后按 LRU 回收）
- `FERMI_DEFAULT_MAX_NEW_TOKENS` 生成默认上限（所有入口统一）
- `FERMI_MAX_NEW_TOKENS_CAP` 生成硬上限
- `FERMI_DEFAULT_TEMPERATURE` 默认温度（范围：`0.0` - `2.0`）
- `FERMI_DEFAULT_TOP_P` 默认 top-p（范围：`(0.0, 1.0]`）
- `FERMI_DEFAULT_REPEAT_PENALTY` 默认重复惩罚（范围：`1.0` - `2.0`）

参数优先级（统一）：
- 请求/CLI 参数 > 环境变量 > `fermi.toml` > 内置默认值

## API 兼容性说明

- OpenAI 接口错误返回统一为 `{"error": {...}}` JSON 结构（不再返回裸字符串）。
- `finish_reason` 语义更准确：
  - 命中 stop token 时返回 `stop`
  - 达到 `max_*_tokens` 上限时返回 `length`

## 模型支持

当前支持：
- **Qwen3**（已测试 `Qwen/Qwen3-1.7B`）

规划中：
- 更多 Qwen / DeepSeek 变体
- 量化权重（更快冷启、更低内存）

## 项目定位

fermi-infer 是一个 **Rust 优先、Mac 友好** 的推理栈：
- 快速启动
- 流式低延迟
- 简单易部署

## 路线图

- KV cache / attention 性能优化
- 吞吐与延迟基准工具
- 更多模型后端
- reasoning 分离（可选）
- 可观测性与指标

## 贡献指南

欢迎 PR，建议流程：
1. Fork 并创建 feature 分支
2. 保持改动集中且清晰
3. 运行 `cargo fmt`（可选 `cargo clippy`）
4. 提交 PR 并附说明/日志
