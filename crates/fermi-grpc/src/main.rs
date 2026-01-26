use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::env;
use std::time::Instant;

use anyhow::{Error as E, Result as AnyResult};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use fermi_grpc::fermi::fermi_server::{Fermi, FermiServer};
use fermi_grpc::fermi::{GenerateRequest, GenerateResponse};
use fermi_io::{download_qwen3_files, load_qwen3_config, load_tokenizer};
use fermi_runtime::{GenerationConfig, InMemorySessionStore, Qwen3Engine, SessionId, SessionStore};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use uuid::Uuid;

#[derive(Clone)]
struct FermiService {
    engine_pool: Vec<Arc<Mutex<Qwen3Engine>>>,
    engine_owner: Arc<Mutex<Vec<Option<SessionId>>>>,
    device: Device,
    tokenizer: Arc<tokenizers::Tokenizer>,
    sessions: Arc<InMemorySessionStore>,
    max_position_embeddings: usize,
    timeout_ms: u64,
    default_system_prompt: Option<String>,
    disable_think: bool,
}

impl FermiService {
    const MAX_NEW_TOKENS: usize = 9056;
    fn new(
        engine_pool: Vec<Arc<Mutex<Qwen3Engine>>>,
        device: Device,
        tokenizer: tokenizers::Tokenizer,
        max_position_embeddings: usize,
        timeout_ms: u64,
        default_system_prompt: Option<String>,
        disable_think: bool,
    ) -> Self {
        let engine_owner = vec![None; engine_pool.len()];
        Self {
            engine_pool,
            engine_owner: Arc::new(Mutex::new(engine_owner)),
            device,
            tokenizer: Arc::new(tokenizer),
            sessions: Arc::new(InMemorySessionStore::new()),
            max_position_embeddings,
            timeout_ms,
            default_system_prompt,
            disable_think,
        }
    }

    fn build_gen_config(&self, req: &GenerateRequest) -> GenerationConfig {
        let mut stop_tokens = Vec::new();
        if let Some(eos) = self.tokenizer.token_to_id("<|endoftext|>") {
            stop_tokens.push(eos);
        }
        if let Some(im_end) = self.tokenizer.token_to_id("<|im_end|>") {
            stop_tokens.push(im_end);
        }

        let mut max_new_tokens = if req.max_new_tokens == 0 {
            256
        } else {
            req.max_new_tokens as usize
        };
        if max_new_tokens > Self::MAX_NEW_TOKENS {
            max_new_tokens = Self::MAX_NEW_TOKENS;
        }
        let repeat_penalty = if req.repeat_penalty <= 0.0 {
            1.1
        } else {
            req.repeat_penalty
        };
        let temperature = if req.temperature <= 0.0 {
            0.8
        } else {
            req.temperature
        };
        let top_p = if req.top_p <= 0.0 {
            0.95
        } else {
            req.top_p
        };

        GenerationConfig {
            max_new_tokens,
            repeat_penalty,
            stop_tokens,
            temperature,
            top_p,
        }
    }

    fn get_engine_for_session(&self, session_id: &SessionId) -> Result<(usize, bool), Status> {
        if let Some(state) = self.sessions.get_state(session_id) {
            if let Some(engine_id) = state.engine_id {
                return Ok((engine_id, false));
            }
        }

        let mut owners = self.engine_owner.lock().map_err(|_| {
            Status::internal("engine owner mutex poisoned")
        })?;
        if let Some((idx, slot)) = owners
            .iter_mut()
            .enumerate()
            .find(|(_, slot)| slot.is_none())
        {
            *slot = Some(session_id.clone());
            self.sessions.update_state(session_id, |state| {
                state.engine_id = Some(idx);
                state.history.clear();
                state.current_pos = 0;
                state.has_context = false;
            });
            return Ok((idx, true));
        }

        Err(Status::resource_exhausted(
            "no idle engine available; increase FERMI_ENGINE_POOL",
        ))
    }

}

fn build_effective_system_prompt(
    req_system: &str,
    default_system: &Option<String>,
    disable_think: bool,
) -> Option<String> {
    let mut sys = if !req_system.is_empty() {
        Some(req_system.to_string())
    } else {
        default_system.clone()
    };
    if disable_think {
        let suffix = "请直接给出最终答案，不输出思考过程，也不要输出<think>标签。";
        sys = Some(match sys {
            Some(s) if !s.is_empty() => format!("{s}\n{suffix}"),
            _ => suffix.to_string(),
        });
    }
    sys
}

fn render_history_prompt(
    pairs: &[(String, String)],
    user_text: &str,
    system_prompt: Option<&str>,
) -> String {
    let mut out = String::new();
    if let Some(sys) = system_prompt {
        if !sys.is_empty() {
            out.push_str("<|im_start|>system\n");
            out.push_str(sys);
            out.push_str("<|im_end|>\n");
        }
    }
    for (user, assistant) in pairs {
        out.push_str("<|im_start|>user\n");
        out.push_str(user);
        out.push_str("<|im_end|>\n<|im_start|>assistant\n");
        out.push_str(assistant);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>user\n");
    out.push_str(user_text);
    out.push_str("<|im_end|>\n<|im_start|>assistant\n");
    out
}

fn render_user_chunk(user_text: &str, has_context: bool, system_prompt: Option<&str>) -> String {
    let mut out = String::new();
    if has_context {
        out.push('\n');
    } else if let Some(sys) = system_prompt {
        if !sys.is_empty() {
            out.push_str("<|im_start|>system\n");
            out.push_str(sys);
            out.push_str("<|im_end|>\n");
        }
    }
    out.push_str("<|im_start|>user\n");
    out.push_str(user_text);
    out.push_str("<|im_end|>\n<|im_start|>assistant\n");
    out
}

fn build_truncated_prompt(
    pairs: &[(String, String)],
    user_text: &str,
    system_prompt: Option<&str>,
    tokenizer: &tokenizers::Tokenizer,
    max_ctx: usize,
    max_new_tokens: usize,
) -> Result<(Vec<u32>, usize), Status> {
    let mut start = 0usize;
    loop {
        let kept = &pairs[start..];
        let prompt = render_history_prompt(kept, user_text, system_prompt);
        let tokens = tokenizer
            .encode(prompt.clone(), false)
            .map_err(|e| Status::internal(e.to_string()))?;
        let input_ids = tokens.get_ids().to_vec();
        let expected_max = input_ids.len() + max_new_tokens + 8;
        if expected_max <= max_ctx || start >= pairs.len() {
            return Ok((input_ids, kept.len()));
        }
        start += 1;
    }
}

#[tonic::async_trait]
impl Fermi for FermiService {
    type GenerateStream =
        Pin<Box<dyn tokio_stream::Stream<Item = std::result::Result<GenerateResponse, Status>> + Send>>;

    async fn generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> std::result::Result<Response<Self::GenerateStream>, Status> {
        let req = request.into_inner();
        let session_id = if req.session_id.is_empty() {
            Uuid::new_v4().to_string()
        } else {
            req.session_id.clone()
        };
        let gen_cfg = self.build_gen_config(&req);
        if !self.sessions.try_begin(&session_id) {
            return Err(Status::resource_exhausted(
                "session is busy; concurrent requests are not supported",
            ));
        }
        let state = self.sessions.get_or_create(session_id.clone());
        let mut history = state.history;
        let mut has_context = state.has_context;
        let system_prompt = build_effective_system_prompt(
            &req.system_prompt,
            &self.default_system_prompt,
            self.disable_think,
        );
        let system_changed = system_prompt.as_deref() != state.system_prompt.as_deref();
        if system_changed {
            history.clear();
            has_context = false;
        }
        let (engine_id, fresh_engine) = self.get_engine_for_session(&session_id)?;
        let mut offset = if has_context { state.current_pos } else { 0 };
        let mut input_ids = self
            .tokenizer
            .encode(
                render_user_chunk(&req.prompt, has_context, system_prompt.as_deref()),
                false,
            )
            .map_err(|e| Status::internal(e.to_string()))?
            .get_ids()
            .to_vec();
        let mut kept_history = history.clone();
        let mut rebuild_cache = fresh_engine || system_changed;
        let expected_max = offset + input_ids.len() + gen_cfg.max_new_tokens + 8;
        if expected_max > self.max_position_embeddings || rebuild_cache {
            let (trunc_ids, kept_pairs) = build_truncated_prompt(
                &history,
                &req.prompt,
                system_prompt.as_deref(),
                &self.tokenizer,
                self.max_position_embeddings,
                gen_cfg.max_new_tokens,
            )?;
            if trunc_ids.len() >= self.max_position_embeddings {
                self.sessions.end(&session_id);
                return Err(Status::invalid_argument(format!(
                    "prompt too long: {} tokens (limit {})",
                    trunc_ids.len(),
                    self.max_position_embeddings
                )));
            }
            input_ids = trunc_ids;
            offset = 0;
            kept_history = if kept_pairs == 0 {
                Vec::new()
            } else {
                history[history.len() - kept_pairs..].to_vec()
            };
            rebuild_cache = true;
        }
        if input_ids.len() >= self.max_position_embeddings {
            self.sessions.end(&session_id);
            return Err(Status::invalid_argument(format!(
                "prompt too long: {} tokens (limit {})",
                input_ids.len(),
                self.max_position_embeddings
            )));
        }

        let (tx, rx) = mpsc::channel(16);
        let engine = Arc::clone(&self.engine_pool[engine_id]);
        let engine_owner = Arc::clone(&self.engine_owner);
        let device = self.device.clone();
        let tokenizer = Arc::clone(&self.tokenizer);
        let sessions = Arc::clone(&self.sessions);
        let session_id_clone: SessionId = session_id.clone();
        let prompt_user = req.prompt.clone();
        let offset_for_cache = offset;
        let input_len = input_ids.len();
        let im_end_id = tokenizer.token_to_id("<|im_end|>");
        let system_prompt_used = system_prompt.clone();
        let mut recent_tokens: Vec<u32> = Vec::with_capacity(12);
        let mut loop_triggered = false;
        let mut timeout_triggered = false;
        let timeout_ms = self.timeout_ms;

        tokio::task::spawn_blocking(move || {
            struct SessionGuard {
                sessions: Arc<InMemorySessionStore>,
                session_id: SessionId,
            }
            impl Drop for SessionGuard {
                fn drop(&mut self) {
                    self.sessions.end(&self.session_id);
                }
            }

            let _guard = SessionGuard {
                sessions: Arc::clone(&sessions),
                session_id: session_id_clone.clone(),
            };
            let mut engine = match engine.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    let _ = tx.blocking_send(Err(Status::internal("engine mutex poisoned")));
                    return;
                }
            };

            let start_time = Instant::now();
            if rebuild_cache {
                engine.clear_kv_cache();
            }
            let mut assistant_buf = String::new();
            let result = engine.generate_stream_with_offset(
                &input_ids,
                offset_for_cache,
                &device,
                &gen_cfg,
                |token_id| {
                    if timeout_ms > 0 && start_time.elapsed().as_millis() as u64 >= timeout_ms {
                        timeout_triggered = true;
                        return Ok(false);
                    }
                    if recent_tokens.len() >= 12 {
                        recent_tokens.remove(0);
                    }
                    recent_tokens.push(token_id);
                    if loop_detected(&recent_tokens) {
                        loop_triggered = true;
                        return Ok(false);
                    }
                    let token_text = tokenizer
                        .decode(&[token_id], true)
                        .map_err(E::msg)?;
                    assistant_buf.push_str(&token_text);
                    let resp = GenerateResponse {
                        token: token_text,
                        token_id,
                        session_id: session_id_clone.clone(),
                    };
                    tx.blocking_send(Ok(resp))
                        .map_err(|e| E::msg(e.to_string()))?;
                    Ok(true)
                },
            );

            if timeout_triggered || loop_triggered {
                engine.clear_kv_cache();
                if let Ok(mut owners) = engine_owner.lock() {
                    owners[engine_id] = None;
                }
                sessions.release(&session_id_clone);
            }

            if timeout_triggered {
                let _ = tx.blocking_send(Err(Status::deadline_exceeded("generation timeout")));
            } else if loop_triggered {
                let _ = tx.blocking_send(Err(Status::aborted("repetitive output detected")));
            } else if let Err(err) = &result {
                let _ = tx.blocking_send(Err(Status::internal(err.to_string())));
            } else {
                if let Ok(generated) = &result {
                    let mut cache_len = offset_for_cache + input_len;
                    if !generated.is_empty() {
                        cache_len += generated.len() - 1;
                        if let Some(&last_token) = generated.last() {
                            if engine.append_tokens(&[last_token], cache_len, &device).is_ok() {
                                cache_len += 1;
                                if let Some(im_end) = im_end_id {
                                    if last_token != im_end {
                                        let _ =
                                            engine.append_tokens(&[im_end], cache_len, &device);
                                        cache_len += 1;
                                    }
                                }
                            }
                        }
                    }
                    kept_history.push((prompt_user, assistant_buf));
                    sessions.update_state(&session_id_clone, |state| {
                        state.history = kept_history;
                        state.has_context = true;
                        state.current_pos = cache_len;
                        state.engine_id = Some(engine_id);
                        state.system_prompt = system_prompt_used.clone();
                    });
                }
                sessions.touch(&session_id_clone);
            }
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx)) as Self::GenerateStream))
    }
}

fn device_setup() -> AnyResult<Device> {
    if cfg!(feature = "cuda") {
        return Ok(Device::new_cuda(0)?);
    } else if cfg!(feature = "metal") {
        return Ok(Device::new_metal(0)?);
    }
    Ok(Device::Cpu)
}

fn env_flag(key: &str) -> bool {
    match env::var(key) {
        Ok(v) => matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"),
        Err(_) => false,
    }
}

fn env_u64(key: &str) -> Option<u64> {
    env::var(key).ok().and_then(|v| v.parse::<u64>().ok())
}

fn loop_detected(recent: &[u32]) -> bool {
    if recent.len() >= 4 {
        let tail = &recent[recent.len() - 4..];
        if tail.iter().all(|&t| t == tail[0]) {
            return true;
        }
    }
    if recent.len() >= 6 {
        let tail = &recent[recent.len() - 6..];
        if tail[0] == tail[2] && tail[2] == tail[4] && tail[1] == tail[3] && tail[3] == tail[5] {
            return true;
        }
    }
    if recent.len() >= 9 {
        let tail = &recent[recent.len() - 9..];
        if tail[0] == tail[3]
            && tail[3] == tail[6]
            && tail[1] == tail[4]
            && tail[4] == tail[7]
            && tail[2] == tail[5]
            && tail[5] == tail[8]
        {
            return true;
        }
    }
    false
}

#[tokio::main]
async fn main() -> AnyResult<()> {
    let device = device_setup()?;
    println!("🚀 gRPC 运行设备: {:?}", device);

    let model_repo_id = env::var("FERMI_MODEL").unwrap_or_else(|_| "Qwen/Qwen3-1.7B".to_string());
    let offline = env_flag("FERMI_OFFLINE") || env_flag("HF_HUB_OFFLINE");
    println!("📥 准备模型文件: {} ...", model_repo_id);
    let files = download_qwen3_files(&model_repo_id, !offline)?;
    println!("⚙️ 正在解析配置文件...");
    let config = load_qwen3_config(&files.config)?;
    let max_position_embeddings = config.max_position_embeddings;

    let dtype = if device.is_metal() { DType::F16 } else { DType::F32 };
    let pool_size = env_u64("FERMI_ENGINE_POOL").unwrap_or(1).max(1) as usize;
    let mut engine_pool = Vec::with_capacity(pool_size);
    for _ in 0..pool_size {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files.weights, dtype, &device)? };
        let mut engine = Qwen3Engine::new(&config, vb)?;
        engine.clear_kv_cache();
        engine_pool.push(Arc::new(Mutex::new(engine)));
    }

    let tokenizer = load_tokenizer(&files.tokenizer)?;

    let addr = "0.0.0.0:50051".parse()?;
    let timeout_ms = env_u64("FERMI_TIMEOUT_MS").unwrap_or(60_000);
    let default_system_prompt = env::var("FERMI_DEFAULT_SYSTEM_PROMPT").ok();
    let disable_think = env_flag("FERMI_DISABLE_THINK");
    let service = FermiService::new(
        engine_pool,
        device,
        tokenizer,
        max_position_embeddings,
        timeout_ms,
        default_system_prompt,
        disable_think,
    );

    println!("✅ gRPC listening on {}", addr);
    tonic::transport::Server::builder()
        .add_service(FermiServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
