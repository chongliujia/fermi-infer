use std::env;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Error as E, Result as AnyResult};
use candle_core::Device;
use fermi_grpc::fermi::fermi_server::{Fermi, FermiServer};
use fermi_grpc::fermi::{GenerateRequest, GenerateResponse};
use fermi_io::load_tokenizer;
use fermi_runtime::{
    GenerationConfig, InMemorySessionStore, InferenceEngine, ModelBuilder, SamplingDefaults,
    SessionId, SessionStore, load_config, resolve_sampling_params, sampling_defaults_from_sources,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use uuid::Uuid;

#[derive(Clone)]
struct FermiService {
    engine_pool: Vec<Arc<Mutex<Box<dyn InferenceEngine>>>>,
    engine_owner: Arc<Mutex<Vec<Option<SessionId>>>>,
    device: Device,
    tokenizer: Arc<tokenizers::Tokenizer>,
    sessions: Arc<InMemorySessionStore>,
    max_position_embeddings: usize,
    timeout_ms: u64,
    sampling_defaults: SamplingDefaults,
    default_system_prompt: Option<String>,
    disable_think: bool,
}

fn release_session_binding(
    engine_owner: &Arc<Mutex<Vec<Option<SessionId>>>>,
    sessions: &Arc<InMemorySessionStore>,
    session_id: &SessionId,
    engine_id: usize,
) {
    if let Ok(mut owners) = engine_owner.lock() {
        owners[engine_id] = None;
    }
    sessions.release(session_id);
}

fn clear_engine_owner(
    owners: &mut [Option<SessionId>],
    session_id: &str,
    preferred_engine_id: Option<usize>,
) {
    if let Some(engine_id) = preferred_engine_id {
        if let Some(slot) = owners.get_mut(engine_id) {
            if slot.as_deref() == Some(session_id) {
                *slot = None;
                return;
            }
        }
    }
    for slot in owners {
        if slot.as_deref() == Some(session_id) {
            *slot = None;
        }
    }
}

impl FermiService {
    const MAX_NEW_TOKENS: usize = 9056;
    fn new(
        engine_pool: Vec<Arc<Mutex<Box<dyn InferenceEngine>>>>,
        device: Device,
        tokenizer: tokenizers::Tokenizer,
        max_position_embeddings: usize,
        timeout_ms: u64,
        sampling_defaults: SamplingDefaults,
        session_ttl_ms: Option<u64>,
        session_max: Option<usize>,
        default_system_prompt: Option<String>,
        disable_think: bool,
    ) -> Self {
        let engine_owner = vec![None; engine_pool.len()];
        Self {
            engine_pool,
            engine_owner: Arc::new(Mutex::new(engine_owner)),
            device,
            tokenizer: Arc::new(tokenizer),
            sessions: Arc::new(InMemorySessionStore::new_with_limits(
                session_ttl_ms.map(Duration::from_millis),
                session_max,
            )),
            max_position_embeddings,
            timeout_ms,
            sampling_defaults,
            default_system_prompt,
            disable_think,
        }
    }

    fn build_gen_config(&self, req: &GenerateRequest) -> Result<GenerationConfig, Status> {
        let mut stop_tokens = Vec::new();
        if let Some(eos) = self.tokenizer.token_to_id("<|endoftext|>") {
            stop_tokens.push(eos);
        }
        if let Some(im_end) = self.tokenizer.token_to_id("<|im_end|>") {
            stop_tokens.push(im_end);
        }

        let requested_max_new_tokens = if req.max_new_tokens == 0 {
            None
        } else {
            Some(req.max_new_tokens as usize)
        };
        // In proto3 scalar fields, omitted values default to 0.
        let requested_temperature = if req.temperature <= 0.0 {
            None
        } else {
            Some(req.temperature)
        };
        let requested_top_p = if req.top_p <= 0.0 {
            None
        } else {
            Some(req.top_p)
        };
        let requested_repeat_penalty = if req.repeat_penalty <= 0.0 {
            None
        } else {
            Some(req.repeat_penalty)
        };

        let mut sampling = resolve_sampling_params(
            requested_max_new_tokens,
            requested_temperature,
            requested_top_p,
            requested_repeat_penalty,
            &self.sampling_defaults,
        )
        .map_err(|err| Status::invalid_argument(err.to_string()))?;
        sampling.max_new_tokens = sampling.max_new_tokens.min(Self::MAX_NEW_TOKENS);

        Ok(GenerationConfig {
            max_new_tokens: sampling.max_new_tokens,
            repeat_penalty: sampling.repeat_penalty,
            stop_tokens,
            temperature: sampling.temperature,
            top_p: sampling.top_p,
        })
    }

    fn get_engine_for_session(&self, session_id: &SessionId) -> Result<(usize, bool), Status> {
        if let Some(state) = self.sessions.get_state(session_id) {
            if let Some(engine_id) = state.engine_id {
                return Ok((engine_id, false));
            }
        }

        let mut owners = self
            .engine_owner
            .lock()
            .map_err(|_| Status::internal("engine owner mutex poisoned"))?;
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

    fn gc_sessions(&self) {
        let evicted = self.sessions.gc_collect();
        if evicted.is_empty() {
            return;
        }
        if let Ok(mut owners) = self.engine_owner.lock() {
            for event in evicted {
                clear_engine_owner(&mut owners, &event.session_id, event.engine_id);
            }
        }
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
    type GenerateStream = Pin<
        Box<dyn tokio_stream::Stream<Item = std::result::Result<GenerateResponse, Status>> + Send>,
    >;

    async fn generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> std::result::Result<Response<Self::GenerateStream>, Status> {
        struct InflightGuard {
            sessions: Arc<InMemorySessionStore>,
            session_id: SessionId,
            enabled: bool,
        }
        impl Drop for InflightGuard {
            fn drop(&mut self) {
                if self.enabled {
                    self.sessions.end(&self.session_id);
                }
            }
        }

        let req = request.into_inner();
        self.gc_sessions();
        let ephemeral_session = req.session_id.is_empty();
        let session_id = if req.session_id.is_empty() {
            Uuid::new_v4().to_string()
        } else {
            req.session_id.clone()
        };
        let gen_cfg = self.build_gen_config(&req)?;
        if !self.sessions.try_begin(&session_id) {
            return Err(Status::resource_exhausted(
                "session is busy; concurrent requests are not supported",
            ));
        }
        let mut inflight_guard = InflightGuard {
            sessions: Arc::clone(&self.sessions),
            session_id: session_id.clone(),
            enabled: true,
        };
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
                if ephemeral_session {
                    if let Ok(mut owners) = self.engine_owner.lock() {
                        owners[engine_id] = None;
                    }
                    self.sessions.release(&session_id);
                }
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
            if ephemeral_session {
                if let Ok(mut owners) = self.engine_owner.lock() {
                    owners[engine_id] = None;
                }
                self.sessions.release(&session_id);
            }
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
        let keep_session = !ephemeral_session;

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
                &mut |token_id| {
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
                    let token_text = tokenizer.decode(&[token_id], true).map_err(E::msg)?;
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
                release_session_binding(&engine_owner, &sessions, &session_id_clone, engine_id);
            }

            if timeout_triggered {
                let _ = tx.blocking_send(Err(Status::deadline_exceeded("generation timeout")));
            } else if loop_triggered {
                let _ = tx.blocking_send(Err(Status::aborted("repetitive output detected")));
            } else if let Err(err) = &result {
                engine.clear_kv_cache();
                release_session_binding(&engine_owner, &sessions, &session_id_clone, engine_id);
                let _ = tx.blocking_send(Err(Status::internal(err.to_string())));
            } else {
                if let Ok(generated) = &result {
                    let mut cache_len = offset_for_cache + input_len;
                    if !generated.is_empty() {
                        cache_len += generated.len() - 1;
                        if let Some(&last_token) = generated.last() {
                            if engine
                                .append_tokens(&[last_token], cache_len, &device)
                                .is_ok()
                            {
                                cache_len += 1;
                                if let Some(im_end) = im_end_id {
                                    if last_token != im_end {
                                        let _ = engine.append_tokens(&[im_end], cache_len, &device);
                                        cache_len += 1;
                                    }
                                }
                            }
                        }
                    }
                    kept_history.push((prompt_user, assistant_buf));
                    if keep_session {
                        sessions.update_state(&session_id_clone, |state| {
                            state.history = kept_history;
                            state.has_context = true;
                            state.current_pos = cache_len;
                            state.engine_id = Some(engine_id);
                            state.system_prompt = system_prompt_used.clone();
                        });
                    }
                }
                if keep_session {
                    sessions.touch(&session_id_clone);
                } else {
                    release_session_binding(&engine_owner, &sessions, &session_id_clone, engine_id);
                }
            }
        });

        inflight_guard.enabled = false;

        Ok(Response::new(
            Box::pin(ReceiverStream::new(rx)) as Self::GenerateStream
        ))
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

fn env_flag_opt(key: &str) -> Option<bool> {
    match env::var(key) {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            if matches!(s.as_str(), "1" | "true" | "yes" | "on") {
                Some(true)
            } else if matches!(s.as_str(), "0" | "false" | "no" | "off") {
                Some(false)
            } else {
                None
            }
        }
        Err(_) => None,
    }
}

fn env_u64(key: &str) -> Option<u64> {
    env::var(key).ok().and_then(|v| v.parse::<u64>().ok())
}

fn resolve_default_system_prompt(
    loaded_cfg: &fermi_runtime::LoadedConfig,
    env_inline: Option<String>,
    env_file: Option<String>,
    cfg_inline: Option<String>,
    cfg_file: Option<String>,
) -> AnyResult<Option<String>> {
    if let Some(prompt) = normalize_prompt_text(env_inline) {
        return Ok(Some(prompt));
    }
    if let Some(path) = normalize_prompt_text(env_file) {
        let prompt = loaded_cfg.read_text_file(&path)?;
        return Ok(normalize_prompt_text(Some(prompt)));
    }
    if let Some(prompt) = normalize_prompt_text(cfg_inline) {
        return Ok(Some(prompt));
    }
    if let Some(path) = normalize_prompt_text(cfg_file) {
        let prompt = loaded_cfg.read_text_file(&path)?;
        return Ok(normalize_prompt_text(Some(prompt)));
    }
    Ok(None)
}

fn normalize_prompt_text(v: Option<String>) -> Option<String> {
    v.and_then(|s| {
        let t = s.trim();
        if t.is_empty() {
            None
        } else {
            Some(t.to_string())
        }
    })
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
    let loaded_cfg = load_config(None)?;
    if let Some(path) = &loaded_cfg.path {
        println!("🧩 使用配置文件: {}", path.display());
    }
    let app_cfg = loaded_cfg.config.clone();

    let device = device_setup()?;
    println!("🚀 gRPC 运行设备: {:?}", device);

    let model_repo_id = env::var("FERMI_MODEL")
        .ok()
        .or_else(|| app_cfg.model.id.clone())
        .unwrap_or_else(|| "Qwen/Qwen3-1.7B".to_string());
    let offline = env_flag_opt("FERMI_OFFLINE")
        .or_else(|| env_flag_opt("HF_HUB_OFFLINE"))
        .or(app_cfg.model.offline)
        .unwrap_or(false);
    println!("📥 准备模型文件: {} ...", model_repo_id);
    let builder = ModelBuilder::new(&model_repo_id, !offline)?;
    let max_position_embeddings = builder.max_position_embeddings();

    let pool_size = env_u64("FERMI_ENGINE_POOL")
        .or_else(|| app_cfg.grpc.engine_pool.and_then(|v| u64::try_from(v).ok()))
        .unwrap_or(1)
        .max(1) as usize;
    let mut engine_pool: Vec<Arc<Mutex<Box<dyn InferenceEngine>>>> = Vec::with_capacity(pool_size);
    for _ in 0..pool_size {
        let mut engine = builder.create_engine(&device)?;
        engine.clear_kv_cache();
        engine_pool.push(Arc::new(Mutex::new(engine)));
    }

    let tokenizer = load_tokenizer(builder.tokenizer_path())?;

    let addr = env::var("FERMI_GRPC_ADDR")
        .ok()
        .or(app_cfg.grpc.addr)
        .unwrap_or_else(|| "0.0.0.0:50051".to_string())
        .parse()?;
    let timeout_ms = env_u64("FERMI_TIMEOUT_MS")
        .or(app_cfg.grpc.timeout_ms)
        .unwrap_or(60_000);
    let sampling_defaults =
        sampling_defaults_from_sources(app_cfg.generation.to_sampling_overrides())?;
    let session_ttl_ms = env_u64("FERMI_SESSION_TTL_MS")
        .or(app_cfg.grpc.session_ttl_ms)
        .filter(|v| *v > 0);
    let session_max = env_u64("FERMI_SESSION_MAX")
        .and_then(|v| usize::try_from(v).ok())
        .or(app_cfg.grpc.session_max)
        .filter(|v| *v > 0);
    let default_system_prompt = resolve_default_system_prompt(
        &loaded_cfg,
        env::var("FERMI_DEFAULT_SYSTEM_PROMPT").ok(),
        env::var("FERMI_DEFAULT_SYSTEM_PROMPT_FILE").ok(),
        app_cfg.grpc.default_system_prompt.clone(),
        app_cfg.grpc.default_system_prompt_file.clone(),
    )?;
    let disable_think = env_flag_opt("FERMI_DISABLE_THINK")
        .or(app_cfg.grpc.disable_think)
        .unwrap_or(false);
    let service = FermiService::new(
        engine_pool,
        device,
        tokenizer,
        max_position_embeddings,
        timeout_ms,
        sampling_defaults,
        session_ttl_ms,
        session_max,
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
