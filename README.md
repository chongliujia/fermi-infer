# Fermi Infer 🚀

![License](https://img.shields.io/badge/license-Apache--2.0-blue) ![Rust](https://img.shields.io/badge/built_with-Rust-orange) ![Platform](https://img.shields.io/badge/platform-macOS-lightgrey)

**The Rust-native inference engine for Small Language Models (SLMs).**
Run efficient models (Qwen, SmolLM, Phi) locally with blazing fast speeds, instant startup times, and full Metal (GPU) acceleration on Apple Silicon.

> **Note**: Currently optimized for **Qwen3**. Support for DeepSeek, Llama, and other architectures is on the roadmap.

## ✨ Why Fermi Infer?

*   **🍎 Mac-First Optimization**: Built on [Candle](https://github.com/huggingface/candle) with native Metal (F16) support.
*   **⚡ Zero-Latency Startup**: No heavy Python runtimes or torch cold-starts. Compiled to a native binary.
*   **🧠 Thinking Mode**: Unique control over reasoning models (like DeepSeek-R1/QwQ), allowing you to toggle the "thought process" on or off via API.
*   **🔌 Drop-in Replacement**: OpenAI-compatible HTTP API ensures it works with your existing tools and frontends.

---

## 🚀 Quick Start

### 1. Interactive CLI Chat
Experience the speed immediately in your terminal.

```bash
# Run with Metal acceleration (macOS)
cargo run -p fermi-infer --release --features metal
```

_Options:_
*   `--features cuda` for Nvidia GPUs.
*   `/reset` to clear context, `/exit` to quit.

### 2. OpenAI-Compatible Server
Serve the model to other apps (like Chatbox, common web UIs, or your own code).

```bash
cargo run -p fermi-openai --release --features metal
# Server listening on http://0.0.0.0:8000
```

---

## 🤖 Supported Models

Fermi Infer focuses on highly optimized pipelines for **efficient small models** running on consumer hardware (Mac M-series, Single RTX 3090).

- [x] **Qwen3**
    - Recommended: `Qwen/Qwen3-1.7B` (Ultra-low latency)
    - Target: 0.5B - 7B variants
- [ ] **DeepSeek-R1 (Distill)**
    - Target: `DeepSeek-R1-Distill-Qwen-1.5B` / `7B`
- [ ] **SmolLM2** (Hugging Face)
    - Target: 135M / 360M / 1.7B - Perfect for ultra-fast local inference
- [ ] **Phi-3.5** (Microsoft)
    - Target: Mini (3.8B) - High reasoning capability
- [ ] **Gemma 2** (Google)
    - Target: 2B / 9B - Efficient lightweight models

*Models are automatically downloaded from HuggingFace on first run.*

---

## 🧠 Feature Spotlight: Thinking Mode

Fermi Infer provides granular control over **Reasoning Models**. You can decide how the model's internal "thought process" is exposed via the OpenAI API.

**Parameter:** `thinking` ("on" | "off" | "auto")

*   **`on`**: The API returns the reasoning trace (e.g., inside `<think>...</think>` tags) alongside the answer.
*   **`off`**: The reasoning is suppressed, returning only the final response.
*   **`auto`**: Intelligent default based on the specific model's capabilities.

**Example Request:**
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{ 
    "model": "Qwen/Qwen3-1.7B",
    "messages": [{"role":"user","content":"Prove that sqrt(2) is irrational."}
    ],
    "thinking": "on",
    "stream": true
  }'
```

---

## 🛠️ Architecture

`fermi-infer` is designed as a modular workspace:

*   **`crates/fermi-runtime`**: The high-performance inference engine (KV-cache, prefill/decode).
*   **`crates/fermi-models`**: Architecture implementations (currently Qwen3).
*   **`crates/fermi-grpc`**: Microservices-ready gRPC streaming server.
*   **`crates/fermi-openai`**: HTTP layer compatible with OpenAI clients.
*   **`crates/fermi-cli`**: The user-facing terminal interface.

## 📦 Configuration

Configure via Environment Variables:

*   `FERMI_MODEL`: Change the model ID (e.g., `export FERMI_MODEL=Qwen/Qwen3-3B`).
*   `FERMI_ENGINE_POOL`: Number of concurrent model instances (HTTP server).
*   `FERMI_OFFLINE=1`: Disable HuggingFace downloads (offline mode).
*   `FERMI_SESSION_TTL_MS`: gRPC session idle TTL in milliseconds (`0`/unset means disabled).
*   `FERMI_SESSION_MAX`: gRPC max in-memory sessions before LRU eviction.

## 🔧 API Compatibility Notes

*   OpenAI-style errors now return `{"error": {...}}` JSON objects (instead of plain strings).
*   `finish_reason` is more precise:
    * `stop` when a stop token is hit.
    * `length` when generation reaches `max_*_tokens`.

## 🗺️ Roadmap

- [ ] Support for DeepSeek-R1 & V3 architectures.
- [ ] KV Cache quantization for lower memory usage.
- [ ] Advanced batching strategies.
- [ ] Production-grade metrics (Prometheus/Grafana).

## 📄 License



Apache-2.0
