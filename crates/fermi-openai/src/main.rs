use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result as AnyResult;
use axum::{
    Json,
    Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, sse::{Event, Sse}},
    routing::{get, post},
};
use axum_extra::TypedHeader;
use axum_extra::headers::Authorization;
use axum_extra::headers::authorization::Bearer;
use candle_core::Device;
use fermi_io::{load_tokenizer};
use fermi_runtime::{GenerationConfig, InferenceEngine, ModelBuilder};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::{mpsc, Semaphore};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tokenizers::Tokenizer;
use tracing::{info, warn};
use uuid::Uuid;

const DEFAULT_MODEL: &str = "Qwen/Qwen3-1.7B";
const DEFAULT_MAX_TOKENS: usize = 256;
const MAX_NEW_TOKENS: usize = 9056;

struct AppState {
    engines: Vec<Arc<Mutex<Box<dyn InferenceEngine>>>>,
    next_engine: AtomicUsize,
    semaphore: Arc<Semaphore>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    stop_tokens: Vec<u32>,
    model_id: String,
    max_position_embeddings: usize,
    disable_think: bool,
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    stream: Option<bool>,
    max_tokens: Option<u32>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    thinking: Option<String>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: serde_json::Value,
}

#[derive(Deserialize)]
struct ResponsesRequest {
    model: Option<String>,
    input: serde_json::Value,
    instructions: Option<String>,
    stream: Option<bool>,
    max_output_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
}

#[derive(Clone, Copy)]
enum ThinkingMode {
    On,
    Off,
    Auto,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct ResponsesResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    output: Vec<ResponseOutput>,
    usage: Usage,
}

#[derive(Serialize)]
struct ResponseOutput {
    id: String,
    r#type: String,
    role: String,
    content: Vec<ResponseContent>,
}

#[derive(Serialize)]
struct ResponseContent {
    r#type: String,
    text: String,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessageOut,
    finish_reason: String,
}

#[derive(Serialize)]
struct ChatMessageOut {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

#[tokio::main]
async fn main() -> AnyResult<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let device = device_setup()?;
    info!("openai server device: {:?}", device);

    let model_id = std::env::var("FERMI_MODEL")
        .ok()
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let offline = env_flag("FERMI_OFFLINE") || env_flag("HF_HUB_OFFLINE");
    let disable_think = env_flag("FERMI_DISABLE_THINK");

    info!("model: {}", model_id);
    
    // Initialize ModelBuilder
    let builder = ModelBuilder::new(&model_id, !offline)?;

    let pool_size = env_u64("FERMI_ENGINE_POOL").unwrap_or(1).max(1) as usize;
    let mut engines: Vec<Arc<Mutex<Box<dyn InferenceEngine>>>> = Vec::with_capacity(pool_size);
    for _ in 0..pool_size {
        let mut engine = builder.create_engine(&device)?;
        engine.clear_kv_cache();
        engines.push(Arc::new(Mutex::new(engine)));
    }

    let tokenizer = Arc::new(load_tokenizer(builder.tokenizer_path())?);
    let stop_tokens = build_stop_tokens(tokenizer.as_ref());
    let semaphore = Arc::new(Semaphore::new(pool_size));

    let state = AppState {
        engines,
        next_engine: AtomicUsize::new(0),
        semaphore,
        tokenizer,
        device,
        stop_tokens,
        model_id: model_id.clone(),
        max_position_embeddings: builder.max_position_embeddings(),
        disable_think,
    };

    let addr = std::env::var("FERMI_OPENAI_ADDR").unwrap_or_else(|_| "0.0.0.0:8000".to_string());
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses))
        .route("/v1/models", get(list_models))
        .with_state(Arc::new(state));

    info!("openai-like server listening on {}", addr);
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
    Ok(())
}

async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let created = unix_ts();
    let body = ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_id.clone(),
            object: "model".to_string(),
            created,
            owned_by: "fermi".to_string(),
        }],
    };
    (StatusCode::OK, Json(body))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    auth: Option<TypedHeader<Authorization<Bearer>>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if auth.is_none() {
        warn!("missing Authorization header");
    }

    let stream = req.stream.unwrap_or(false);
    let max_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .map(|v| v as usize)
        .unwrap_or(DEFAULT_MAX_TOKENS)
        .min(MAX_NEW_TOKENS);
    let temperature = req.temperature.unwrap_or(0.8);
    let top_p = req.top_p.unwrap_or(0.95);

    let requested = req.thinking.as_deref();
    let default_mode = env_default_thinking();
    let mut think_mode = match requested {
        Some(v) => parse_thinking_mode(v),
        None => default_mode,
    };
    if state.disable_think {
        think_mode = ThinkingMode::Off;
    } else if matches!(think_mode, ThinkingMode::On | ThinkingMode::Auto) {
        let model_id = req.model.as_deref().unwrap_or(&state.model_id);
        if !supports_thinking(model_id) {
            think_mode = ThinkingMode::Off;
        } else if matches!(think_mode, ThinkingMode::Auto) {
            think_mode = ThinkingMode::On;
        }
    }

    let prompt = build_prompt(&req.messages, think_mode);
    let prompt_tokens = match state.tokenizer.encode(prompt.clone(), false) {
        Ok(tokens) => tokens.get_ids().len(),
        Err(err) => {
            return (StatusCode::BAD_REQUEST, err.to_string()).into_response();
        }
    };

    if prompt_tokens + max_tokens + 8 > state.max_position_embeddings {
        let msg = format!(
            "prompt too long: {} tokens (limit {})",
            prompt_tokens,
            state.max_position_embeddings
        );
        return (StatusCode::BAD_REQUEST, msg).into_response();
    }

    let gen_cfg = GenerationConfig {
        max_new_tokens: max_tokens,
        repeat_penalty: 1.1,
        stop_tokens: state.stop_tokens.clone(),
        temperature,
        top_p,
    };

    if stream {
        return stream_chat(state, prompt, gen_cfg).await;
    }

    let reply = match run_inference(state.clone(), prompt, gen_cfg).await {
        Ok(r) => r,
        Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    };

    let created = unix_ts();
    let body = ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created,
        model: req.model.unwrap_or_else(|| state.model_id.clone()),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageOut {
                role: "assistant".to_string(),
                content: reply.text,
            },
            finish_reason: reply.finish_reason,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: reply.completion_tokens,
            total_tokens: prompt_tokens + reply.completion_tokens,
        },
    };

    (StatusCode::OK, Json(body)).into_response()
}

async fn responses(
    State(state): State<Arc<AppState>>,
    auth: Option<TypedHeader<Authorization<Bearer>>>,
    Json(req): Json<ResponsesRequest>,
) -> Response {
    if auth.is_none() {
        warn!("missing Authorization header");
    }

    let stream = req.stream.unwrap_or(false);
    let max_tokens = req
        .max_output_tokens
        .map(|v| v as usize)
        .unwrap_or(DEFAULT_MAX_TOKENS)
        .min(MAX_NEW_TOKENS);
    let temperature = req.temperature.unwrap_or(0.8);
    let top_p = req.top_p.unwrap_or(0.95);

    let messages = match normalize_responses_input(&req.input, req.instructions.as_deref()) {
        Ok(m) => m,
        Err(err) => return (StatusCode::BAD_REQUEST, err).into_response(),
    };
    let prompt = build_prompt(&messages, ThinkingMode::Off);

    let prompt_tokens = match state.tokenizer.encode(prompt.clone(), false) {
        Ok(tokens) => tokens.get_ids().len(),
        Err(err) => return (StatusCode::BAD_REQUEST, err.to_string()).into_response(),
    };
    if prompt_tokens + max_tokens + 8 > state.max_position_embeddings {
        let msg = format!(
            "prompt too long: {} tokens (limit {})",
            prompt_tokens,
            state.max_position_embeddings
        );
        return (StatusCode::BAD_REQUEST, msg).into_response();
    }

    let gen_cfg = GenerationConfig {
        max_new_tokens: max_tokens,
        repeat_penalty: 1.1,
        stop_tokens: state.stop_tokens.clone(),
        temperature,
        top_p,
    };

    if stream {
        return stream_responses(state, prompt, gen_cfg).await;
    }

    let reply = match run_inference(state.clone(), prompt, gen_cfg).await {
        Ok(r) => r,
        Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    };

    let created = unix_ts();
    let body = ResponsesResponse {
        id: format!("resp-{}", Uuid::new_v4()),
        object: "response".to_string(),
        created,
        model: req.model.unwrap_or_else(|| state.model_id.clone()),
        output: vec![ResponseOutput {
            id: format!("msg-{}", Uuid::new_v4()),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![ResponseContent {
                r#type: "output_text".to_string(),
                text: reply.text,
            }],
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: reply.completion_tokens,
            total_tokens: prompt_tokens + reply.completion_tokens,
        },
    };

    (StatusCode::OK, Json(body)).into_response()
}

struct InferenceResult {
    text: String,
    completion_tokens: usize,
    finish_reason: String,
}

async fn run_inference(state: Arc<AppState>, prompt: String, cfg: GenerationConfig) -> AnyResult<InferenceResult> {
    let permit = state.semaphore.clone().acquire_owned().await?;
    let engine = next_engine(&state);
    let tokenizer = state.tokenizer.clone();
    let device = state.device.clone();

    let result = tokio::task::spawn_blocking(move || -> AnyResult<InferenceResult> {
        let _permit = permit;
        let mut engine = engine.lock().expect("engine mutex poisoned");
        engine.clear_kv_cache();
        let tokens = tokenizer.encode(prompt, false).map_err(anyhow::Error::msg)?;
        let input_ids = tokens.get_ids().to_vec();

        let mut utf8 = Utf8Buffer::new();
        let mut out = String::new();
        let generated = engine.generate_stream(&input_ids, &device, &cfg, &mut |token_id| {
            if let Some(text) = utf8.push_and_decode(token_id, &tokenizer)? {
                out.push_str(&text);
            }
            Ok(true)
        })?;
        if let Some(tail) = utf8.flush(&tokenizer)? {
            out.push_str(&tail);
        }

        let finish_reason = if !generated.is_empty() {
            "stop".to_string()
        } else {
            "length".to_string()
        };

        Ok(InferenceResult {
            text: out,
            completion_tokens: generated.len(),
            finish_reason,
        })
    }).await??;

    Ok(result)
}

async fn stream_chat(state: Arc<AppState>, prompt: String, cfg: GenerationConfig) -> Response {
    let permit = match state.semaphore.clone().acquire_owned().await {
        Ok(p) => p,
        Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    };
    let engine = next_engine(&state);
    let tokenizer = state.tokenizer.clone();
    let device = state.device.clone();
    let model = state.model_id.clone();
    let created = unix_ts();

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);

    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut engine = engine.lock().expect("engine mutex poisoned");
        engine.clear_kv_cache();

        let id = format!("chatcmpl-{}", Uuid::new_v4());
        let start_evt = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}],
        });
        let _ = tx.blocking_send(Ok(Event::default().data(start_evt.to_string())));

        let tokens = match tokenizer.encode(prompt, false) {
            Ok(t) => t,
            Err(err) => {
                let _ = tx.blocking_send(Ok(Event::default().data(err.to_string())));
                return;
            }
        };
        let input_ids = tokens.get_ids().to_vec();
        let mut utf8 = Utf8Buffer::new();

        let result = engine.generate_stream(&input_ids, &device, &cfg, &mut |token_id| {
            if let Some(text) = utf8.push_and_decode(token_id, &tokenizer)? {
                let chunk = json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}],
                });
                let _ = tx.blocking_send(Ok(Event::default().data(chunk.to_string())));
            }
            Ok(true)
        });

        if let Ok(tail) = utf8.flush(&tokenizer) {
            if let Some(text) = tail {
                let chunk = json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}],
                });
                let _ = tx.blocking_send(Ok(Event::default().data(chunk.to_string())));
            }
        }

        let finish_reason = match result {
            Ok(_) => "stop",
            Err(_) => "error",
        };
        let end_evt = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        });
        let _ = tx.blocking_send(Ok(Event::default().data(end_evt.to_string())));
        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    let stream = ReceiverStream::new(rx).map(|evt| evt);
    Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

async fn stream_responses(state: Arc<AppState>, prompt: String, cfg: GenerationConfig) -> Response {
    let permit = match state.semaphore.clone().acquire_owned().await {
        Ok(p) => p,
        Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    };
    let engine = next_engine(&state);
    let tokenizer = state.tokenizer.clone();
    let device = state.device.clone();
    let model = state.model_id.clone();
    let created = unix_ts();

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);

    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut engine = engine.lock().expect("engine mutex poisoned");
        engine.clear_kv_cache();

        let id = format!("resp-{}", Uuid::new_v4());
        let start_evt = json!({
            "id": id,
            "object": "response",
            "created": created,
            "model": model,
            "type": "response.created"
        });
        let _ = tx.blocking_send(Ok(Event::default().data(start_evt.to_string())));

        let tokens = match tokenizer.encode(prompt, false) {
            Ok(t) => t,
            Err(err) => {
                let _ = tx.blocking_send(Ok(Event::default().data(err.to_string())));
                return;
            }
        };
        let input_ids = tokens.get_ids().to_vec();
        let mut utf8 = Utf8Buffer::new();

        let result = engine.generate_stream(&input_ids, &device, &cfg, &mut |token_id| {
            if let Some(text) = utf8.push_and_decode(token_id, &tokenizer)? {
                let chunk = json!({
                    "id": id,
                    "object": "response",
                    "created": created,
                    "model": model,
                    "type": "response.output_text.delta",
                    "delta": text
                });
                let _ = tx.blocking_send(Ok(Event::default().data(chunk.to_string())));
            }
            Ok(true)
        });

        if let Ok(tail) = utf8.flush(&tokenizer) {
            if let Some(text) = tail {
                let chunk = json!({
                    "id": id,
                    "object": "response",
                    "created": created,
                    "model": model,
                    "type": "response.output_text.delta",
                    "delta": text
                });
                let _ = tx.blocking_send(Ok(Event::default().data(chunk.to_string())));
            }
        }

        let finish_type = match result {
            Ok(_) => "response.completed",
            Err(_) => "response.failed",
        };
        let end_evt = json!({
            "id": id,
            "object": "response",
            "created": created,
            "model": model,
            "type": finish_type
        });
        let _ = tx.blocking_send(Ok(Event::default().data(end_evt.to_string())));
        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    let stream = ReceiverStream::new(rx).map(|evt| evt);
    Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

fn build_prompt(messages: &[ChatMessage], think_mode: ThinkingMode) -> String {
    let mut out = String::new();

    for msg in messages {
        let role = msg.role.as_str();
        let content = match extract_content(&msg.content) {
            Some(c) => c,
            None => continue,
        };
        match role {
            "system" | "developer" => {
                out.push_str("<|im_start|>system\n");
                out.push_str(&content);
                out.push_str("<|im_end|>\n");
            }
            "user" => {
                out.push_str("<|im_start|>user\n");
                out.push_str(&content);
                out.push_str("<|im_end|>\n");
            }
            "assistant" => {
                out.push_str("<|im_start|>assistant\n");
                out.push_str(&content);
                out.push_str("<|im_end|>\n");
            }
            _ => {}
        }
    }

    match think_mode {
        ThinkingMode::Off => {
            out.push_str("<|im_start|>system\n");
            out.push_str("请直接给出最终答案，不输出思考过程，也不要输出<think>标签。\n");
            out.push_str("<|im_end|>\n");
        }
        ThinkingMode::On => {
            out.push_str("<|im_start|>system\n");
            out.push_str("请将思考过程放在<think>...</think>中，最终答案放在think之外。\n");
            out.push_str("<|im_end|>\n");
        }
        ThinkingMode::Auto => {}
    }

    out.push_str("<|im_start|>assistant\n");
    out
}

fn normalize_responses_input(
    input: &serde_json::Value,
    instructions: Option<&str>,
) -> Result<Vec<ChatMessage>, String> {
    let mut out = Vec::new();
    if let Some(instr) = instructions {
        if !instr.is_empty() {
            out.push(ChatMessage {
                role: "system".to_string(),
                content: serde_json::Value::String(instr.to_string()),
            });
        }
    }
    match input {
        serde_json::Value::String(s) => {
            out.push(ChatMessage {
                role: "user".to_string(),
                content: serde_json::Value::String(s.clone()),
            });
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                if let Some(role) = item.get("role").and_then(|v| v.as_str()) {
                    let content = item.get("content").cloned().unwrap_or(serde_json::Value::Null);
                    out.push(ChatMessage {
                        role: role.to_string(),
                        content,
                    });
                } else if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                    out.push(ChatMessage {
                        role: "user".to_string(),
                        content: serde_json::Value::String(text.to_string()),
                    });
                }
            }
        }
        _ => return Err("input must be string or array".to_string()),
    }
    if out.is_empty() {
        return Err("input is empty".to_string());
    }
    Ok(out)
}

fn extract_content(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(arr) => {
            let mut parts = Vec::new();
            for item in arr {
                if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                    parts.push(text.to_string());
                }
            }
            if parts.is_empty() {
                None
            } else {
                Some(parts.join(""))
            }
        }
        _ => None,
    }
}

fn next_engine(state: &AppState) -> Arc<Mutex<Box<dyn InferenceEngine>>> {
    let idx = state.next_engine.fetch_add(1, Ordering::Relaxed) % state.engines.len();
    Arc::clone(&state.engines[idx])
}

fn build_stop_tokens(tokenizer: &Tokenizer) -> Vec<u32> {
    let mut stop_tokens = Vec::new();
    if let Some(eos) = tokenizer.token_to_id("<|endoftext|>") {
        stop_tokens.push(eos);
    }
    if let Some(im_end) = tokenizer.token_to_id("<|im_end|>") {
        stop_tokens.push(im_end);
    }
    stop_tokens
}

fn device_setup() -> AnyResult<Device> {
    if cfg!(feature = "cuda") {
        return Ok(Device::new_cuda(0)?);
    } else if cfg!(feature = "metal") {
        return Ok(Device::new_metal(0)?);
    }
    Ok(Device::Cpu)
}

fn unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn env_flag(key: &str) -> bool {
    match std::env::var(key) {
        Ok(v) => matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"),
        Err(_) => false,
    }
}

fn parse_thinking_mode(value: &str) -> ThinkingMode {
    match value.to_ascii_lowercase().as_str() {
        "on" | "true" | "1" | "yes" => ThinkingMode::On,
        "off" | "false" | "0" | "no" => ThinkingMode::Off,
        "auto" => ThinkingMode::Auto,
        _ => ThinkingMode::Auto,
    }
}

fn env_default_thinking() -> ThinkingMode {
    match std::env::var("FERMI_DEFAULT_THINKING") {
        Ok(v) => parse_thinking_mode(&v),
        Err(_) => ThinkingMode::Off,
    }
}

fn supports_thinking(model_id: &str) -> bool {
    if let Ok(v) = std::env::var("FERMI_SUPPORTS_THINKING") {
        return matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on");
    }
    let id = model_id.to_ascii_lowercase();
    id.contains("qwen") || id.contains("deepseek") || id.contains("r1") || id.contains("qwq")
}

fn env_u64(key: &str) -> Option<u64> {
    std::env::var(key).ok().and_then(|v| v.parse::<u64>().ok())
}

struct Utf8Buffer {
    pending_ids: Vec<u32>,
}

impl Utf8Buffer {
    fn new() -> Self {
        Self {
            pending_ids: Vec::new(),
        }
    }

    fn push_and_decode(&mut self, token_id: u32, tokenizer: &Tokenizer) -> AnyResult<Option<String>> {
        self.pending_ids.push(token_id);
        let text = tokenizer.decode(&self.pending_ids, true).map_err(anyhow::Error::msg)?;
        if text.contains('\u{FFFD}') {
            Ok(None)
        } else {
            self.pending_ids.clear();
            Ok(Some(text))
        }
    }

    fn flush(&mut self, tokenizer: &Tokenizer) -> AnyResult<Option<String>> {
        if self.pending_ids.is_empty() {
            return Ok(None);
        }
        let text = tokenizer.decode(&self.pending_ids, true).map_err(anyhow::Error::msg)?;
        self.pending_ids.clear();
        Ok(Some(text))
    }
}