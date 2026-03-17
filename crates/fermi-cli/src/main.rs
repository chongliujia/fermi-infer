use anyhow::{Error as E, Result};
use candle_core::Device;
use fermi_io::{ModelArch, load_tokenizer};
use fermi_runtime::{
    GenerationConfig, ModelBuilder, assistant_turn_end_token_id, apply_sampling_preset,
    build_stop_tokens, load_config, parse_sampling_preset, render_history_prompt,
    render_user_chunk, resolve_sampling_params, sampling_defaults_from_sources,
};
use std::env;
use std::io::{self, Write};
use std::time::Instant;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let cli_cfg = parse_args()?;
    let loaded_cfg = load_config(cli_cfg.config.as_deref())?;
    if let Some(path) = &loaded_cfg.path {
        println!("🧩 配置文件: {}", path.display());
    }
    let app_cfg = loaded_cfg.config.clone();
    // 1. 基础环境设置
    let device = device_setup()?;
    println!("🚀 运行设备: {:?}", device);

    // ==========================================
    // 指定 Qwen3 官方模型 ID / 本地路径
    // ==========================================
    let (model_repo_id, model_source) = resolve_model_id(
        cli_cfg.model.clone(),
        std::env::var("FERMI_MODEL").ok(),
        app_cfg.model.id.clone(),
        "Qwen/Qwen3-1.7B",
    );
    let offline = cli_cfg
        .offline
        .or_else(|| env_flag_opt("FERMI_OFFLINE"))
        .or_else(|| env_flag_opt("HF_HUB_OFFLINE"))
        .or(app_cfg.model.offline)
        .unwrap_or(false);

    println!("📥 准备模型文件...");
    println!("📦 模型: {} ({})", model_repo_id, model_source);

    let builder = ModelBuilder::new(&model_repo_id, !offline)?;
    println!("🧠 架构: {:?}", builder.model_arch());
    let mut sampling_defaults =
        sampling_defaults_from_sources(app_cfg.generation.to_sampling_overrides())?;
    if let Some(preset) = cli_cfg.preset.as_deref() {
        sampling_defaults =
            apply_sampling_preset(&sampling_defaults, builder.model_arch(), parse_sampling_preset(preset)?);
    }
    let sampling = resolve_sampling_params(
        cli_cfg.max_new_tokens,
        cli_cfg.temperature,
        cli_cfg.top_p,
        cli_cfg.repeat_penalty,
        &sampling_defaults,
    )?;

    println!("✅ 权重下载/验证完成");
    println!("⚙️ 正在初始化推理引擎...");

    let mut engine = builder.create_engine(&device)?;
    let tokenizer = load_tokenizer(builder.tokenizer_path())?;
    let model_arch = builder.model_arch();
    engine.clear_kv_cache();

    let mut history: Vec<(String, String)> = Vec::new();
    let mut current_pos: usize = 0;
    let mut has_context = false;
    let turn_end_id = assistant_turn_end_token_id(model_arch, &tokenizer);

    let gen_cfg = GenerationConfig {
        max_new_tokens: sampling.max_new_tokens,
        repeat_penalty: sampling.repeat_penalty,
        stop_tokens: build_stop_tokens(model_arch, &tokenizer),
        temperature: sampling.temperature,
        top_p: sampling.top_p,
    };
    let max_ctx = builder.max_position_embeddings();
    let timeout_ms = cli_cfg
        .timeout_ms
        .or_else(|| env_u64("FERMI_TIMEOUT_MS"))
        .or(app_cfg.cli.timeout_ms)
        .unwrap_or(60_000);
    let disable_think = env_flag_opt("FERMI_DISABLE_THINK")
        .or(app_cfg.cli.disable_think)
        .unwrap_or(false);
    let mut default_system_prompt = resolve_default_system_prompt(
        cli_cfg.system_prompt.clone(),
        cli_cfg.system_prompt_file.clone(),
        env::var("FERMI_DEFAULT_SYSTEM_PROMPT").ok(),
        env::var("FERMI_DEFAULT_SYSTEM_PROMPT_FILE").ok(),
        app_cfg.cli.default_system_prompt.clone(),
        app_cfg.cli.default_system_prompt_file.clone(),
        &loaded_cfg,
    )?;
    if disable_think {
        default_system_prompt = Some(append_disable_think_hint(default_system_prompt.as_deref()));
    }

    if let RunMode::SinglePrompt(prompt) = &cli_cfg.run_mode {
        let result = generate_single_response(
            &mut *engine,
            &tokenizer,
            &device,
            model_arch,
            prompt,
            default_system_prompt.as_deref(),
            &gen_cfg,
            timeout_ms,
        )?;
        println!("{}", result.text);
        if let Some(reason) = result.truncated_reason {
            eprintln!("⚠️ 输出已提前停止: {}", reason);
        }
        println!("\n\n✅ 完成");
        return Ok(());
    }

    println!("💬 交互模式：输入问题，回车发送。命令：/help /reset /exit");

    loop {
        print!("> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match line {
            "/exit" | "/quit" => break,
            "/reset" => {
                history.clear();
                engine.clear_kv_cache();
                current_pos = 0;
                has_context = false;
                println!("✅ 已清空上下文");
                continue;
            }
            "/help" => {
                println!("命令：/help /reset /exit");
                continue;
            }
            _ => {}
        }

        let mut offset = if has_context { current_pos } else { 0 };
        let mut input_ids = tokenizer
            .encode(
                render_user_chunk(model_arch, line, has_context, default_system_prompt.as_deref()),
                false,
            )
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let expected_max = offset + input_ids.len() + gen_cfg.max_new_tokens + 8;
        if expected_max > max_ctx {
            let pairs = history_pairs(&history);
            let (trunc_ids, kept_pairs) = build_truncated_prompt(
                model_arch,
                &pairs,
                line,
                default_system_prompt.as_deref(),
                &tokenizer,
                max_ctx,
                gen_cfg.max_new_tokens,
            )?;
            if trunc_ids.len() >= max_ctx {
                println!("⚠️ 输入过长，已超过最大上下文 {} tokens", max_ctx);
                continue;
            }
            engine.clear_kv_cache();
            current_pos = 0;
            offset = 0;
            has_context = kept_pairs > 0;
            input_ids = trunc_ids;
            if kept_pairs < pairs.len() {
                println!("⚠️ 上下文过长，已自动截断");
            } else {
                println!("⚠️ 上下文过长，已自动重建缓存");
            }
        }
        if input_ids.len() >= max_ctx {
            println!("⚠️ 输入过长，已超过最大上下文 {} tokens", max_ctx);
            continue;
        }

        let mut assistant_buf = String::new();
        let mut utf8_buffer = Utf8Buffer::new();
        let mut think_filter = ThinkFilter::new();

        // Loop detection state
        let mut recent_tokens: Vec<u32> = Vec::with_capacity(12);
        let mut loop_triggered = false;
        let mut timeout_triggered = false;
        let start_time = Instant::now();

        // Pass a mutable closure to the trait method
        let generated = engine.generate_stream_with_offset(
            &input_ids,
            offset,
            &device,
            &gen_cfg,
            &mut |token_id| {
                if timeout_ms > 0 && start_time.elapsed().as_millis() as u64 >= timeout_ms {
                    timeout_triggered = true;
                    return Ok(false);
                }
                // Loop detection logic
                if recent_tokens.len() >= 12 {
                    recent_tokens.remove(0);
                }
                recent_tokens.push(token_id);
                if loop_detected(&recent_tokens) {
                    loop_triggered = true;
                    return Ok(false);
                }

                if let Some(text) = utf8_buffer.push_and_decode(token_id, &tokenizer)? {
                    let filtered = think_filter.process(&text);
                    if !filtered.is_empty() {
                        assistant_buf.push_str(&filtered);
                        print!("{}", filtered);
                        io::stdout().flush()?;
                    }
                }
                Ok(true)
            },
        )?;
        if let Some(tail_text) = utf8_buffer.flush(&tokenizer)? {
            let filtered = think_filter.process(&tail_text);
            if !filtered.is_empty() {
                assistant_buf.push_str(&filtered);
                print!("{}", filtered);
                io::stdout().flush()?;
            }
        }
        let tail = think_filter.flush();
        if !tail.is_empty() {
            assistant_buf.push_str(&tail);
            print!("{}", tail);
            io::stdout().flush()?;
        }
        println!();
        if timeout_triggered || loop_triggered {
            engine.clear_kv_cache();
            history.clear();
            current_pos = 0;
            has_context = false;
            if timeout_triggered {
                println!("\n⚠️ 生成超时，已重置上下文");
            } else {
                println!("\n⚠️ 检测到重复输出，已重置上下文");
            }
            continue;
        }

        history.push(("user".to_string(), line.to_string()));
        history.push(("assistant".to_string(), assistant_buf));

        let mut cache_len = current_pos + input_ids.len();
        if !generated.is_empty() {
            cache_len += generated.len() - 1;
            if let Some(&last_token) = generated.last() {
                engine.append_tokens(&[last_token], cache_len, &device)?;
                cache_len += 1;
                if let Some(turn_end) = turn_end_id {
                    if last_token != turn_end {
                        engine.append_tokens(&[turn_end], cache_len, &device)?;
                        cache_len += 1;
                    }
                }
            }
        }
        current_pos = cache_len;
        has_context = true;
    }

    println!("\n\n✅ 完成");
    Ok(())
}

fn device_setup() -> Result<Device> {
    if cfg!(feature = "cuda") {
        return Ok(Device::new_cuda(0)?);
    } else if cfg!(feature = "metal") {
        return Ok(Device::new_metal(0)?);
    }
    Ok(Device::Cpu)
}

struct CliConfig {
    max_new_tokens: Option<usize>,
    repeat_penalty: Option<f32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    model: Option<String>,
    offline: Option<bool>,
    config: Option<String>,
    timeout_ms: Option<u64>,
    system_prompt: Option<String>,
    system_prompt_file: Option<String>,
    run_mode: RunMode,
    preset: Option<String>,
}

#[derive(Debug, Clone)]
enum RunMode {
    Chat,
    SinglePrompt(String),
}

fn parse_args() -> Result<CliConfig> {
    let mut max_new_tokens = None;
    let mut repeat_penalty = None;
    let mut temperature = None;
    let mut top_p = None;
    let mut model: Option<String> = None;
    let mut offline: Option<bool> = None;
    let mut config: Option<String> = None;
    let mut timeout_ms: Option<u64> = None;
    let mut system_prompt: Option<String> = None;
    let mut system_prompt_file: Option<String> = None;
    let mut run_mode = RunMode::Chat;
    let mut preset: Option<String> = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "chat" => {}
            "--model" => {
                if let Some(v) = args.next() {
                    model = Some(v);
                } else {
                    return Err(E::msg("--model requires a value"));
                }
            }
            "--offline" => {
                offline = Some(true);
            }
            "--online" => {
                offline = Some(false);
            }
            "--config" => {
                if let Some(v) = args.next() {
                    config = Some(v);
                } else {
                    return Err(E::msg("--config requires a value"));
                }
            }
            "--prompt" => {
                if let Some(v) = args.next() {
                    run_mode = RunMode::SinglePrompt(v);
                } else {
                    return Err(E::msg("--prompt requires a value"));
                }
            }
            "--system-prompt" => {
                if let Some(v) = args.next() {
                    system_prompt = Some(v);
                } else {
                    return Err(E::msg("--system-prompt requires a value"));
                }
            }
            "--system-prompt-file" => {
                if let Some(v) = args.next() {
                    system_prompt_file = Some(v);
                } else {
                    return Err(E::msg("--system-prompt-file requires a value"));
                }
            }
            "--preset" => {
                if let Some(v) = args.next() {
                    preset = Some(v);
                } else {
                    return Err(E::msg("--preset requires a value"));
                }
            }
            "--timeout-ms" => {
                if let Some(v) = args.next() {
                    timeout_ms = Some(v.parse::<u64>().map_err(E::msg)?);
                } else {
                    return Err(E::msg("--timeout-ms requires a value"));
                }
            }
            "--max-new-tokens" => {
                if let Some(v) = args.next() {
                    max_new_tokens = Some(v.parse::<usize>().map_err(E::msg)?);
                } else {
                    return Err(E::msg("--max-new-tokens requires a value"));
                }
            }
            "--repeat-penalty" => {
                if let Some(v) = args.next() {
                    repeat_penalty = Some(v.parse::<f32>().map_err(E::msg)?);
                } else {
                    return Err(E::msg("--repeat-penalty requires a value"));
                }
            }
            "--temperature" => {
                if let Some(v) = args.next() {
                    temperature = Some(v.parse::<f32>().map_err(E::msg)?);
                } else {
                    return Err(E::msg("--temperature requires a value"));
                }
            }
            "--top-p" => {
                if let Some(v) = args.next() {
                    top_p = Some(v.parse::<f32>().map_err(E::msg)?);
                } else {
                    return Err(E::msg("--top-p requires a value"));
                }
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                return Err(E::msg(format!("unknown argument: {}", other)));
            }
        }
    }

    Ok(CliConfig {
        max_new_tokens,
        repeat_penalty,
        temperature,
        top_p,
        model,
        offline,
        config,
        timeout_ms,
        system_prompt,
        system_prompt_file,
        run_mode,
        preset,
    })
}

fn print_usage() {
    println!(
        "Usage: fermi-infer [chat] [--config PATH] [--model ID|PATH] [--prompt TEXT] [--system-prompt TEXT] [--system-prompt-file PATH] [--preset NAME] [--offline|--online] [--timeout-ms MS] [--max-new-tokens N] [--repeat-penalty P] [--temperature T] [--top-p P]"
    );
    println!("  chat              Explicitly start interactive chat mode (default if omitted)");
    println!("  --config          Config file path (default auto-discover: ./fermi.toml)");
    println!(
        "  --model           HuggingFace repo id or local model dir (default: Qwen/Qwen3-1.7B)"
    );
    println!("  --prompt          Run a single prompt and exit");
    println!("  --system-prompt   Override the default system prompt with inline text");
    println!("  --system-prompt-file  Load the system prompt from a file");
    println!("  --preset          Sampling preset: chat-balanced, chat-precise, reasoning, creative");
    println!("  --offline         Disable network access; require local model files");
    println!("  --online          Force enable network access");
    println!(
        "  --timeout-ms      Per-request timeout in milliseconds (default: 60000; 0 disables)"
    );
    println!("  --max-new-tokens  Maximum number of generated tokens");
    println!("  --repeat-penalty  Repetition penalty in [1.0, 2.0]");
    println!("  --temperature     Sampling temperature in [0.0, 2.0]");
    println!("  --top-p           Nucleus sampling p in (0.0, 1.0]");
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
    cli_inline: Option<String>,
    cli_file: Option<String>,
    env_inline: Option<String>,
    env_file: Option<String>,
    cfg_inline: Option<String>,
    cfg_file: Option<String>,
    loaded_cfg: &fermi_runtime::LoadedConfig,
) -> Result<Option<String>> {
    if let Some(prompt) = normalize_prompt_text(cli_inline) {
        return Ok(Some(prompt));
    }
    if let Some(path) = normalize_prompt_text(cli_file) {
        let prompt = loaded_cfg.read_text_file(&path).map_err(E::msg)?;
        return Ok(normalize_prompt_text(Some(prompt)));
    }
    if let Some(prompt) = normalize_prompt_text(env_inline) {
        return Ok(Some(prompt));
    }
    if let Some(path) = normalize_prompt_text(env_file) {
        let prompt = loaded_cfg.read_text_file(&path).map_err(E::msg)?;
        return Ok(normalize_prompt_text(Some(prompt)));
    }
    if let Some(prompt) = normalize_prompt_text(cfg_inline) {
        return Ok(Some(prompt));
    }
    if let Some(path) = normalize_prompt_text(cfg_file) {
        let prompt = loaded_cfg.read_text_file(&path).map_err(E::msg)?;
        return Ok(normalize_prompt_text(Some(prompt)));
    }
    Ok(None)
}

fn resolve_model_id(
    cli_model: Option<String>,
    env_model: Option<String>,
    cfg_model: Option<String>,
    default_model: &str,
) -> (String, &'static str) {
    if let Some(model) = cli_model {
        return (model, "source=cli");
    }
    if let Some(model) = env_model {
        return (model, "source=env");
    }
    if let Some(model) = cfg_model {
        return (model, "source=config");
    }
    (default_model.to_string(), "source=default")
}

fn generate_single_response(
    engine: &mut dyn fermi_runtime::InferenceEngine,
    tokenizer: &Tokenizer,
    device: &Device,
    model_arch: ModelArch,
    prompt: &str,
    system_prompt: Option<&str>,
    gen_cfg: &GenerationConfig,
    timeout_ms: u64,
) -> Result<SingleResponseResult> {
    let input_ids = tokenizer
        .encode(render_user_chunk(model_arch, prompt, false, system_prompt), false)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let mut assistant_buf = String::new();
    let mut utf8_buffer = Utf8Buffer::new();
    let mut think_filter = ThinkFilter::new();
    let mut recent_tokens: Vec<u32> = Vec::with_capacity(12);
    let mut loop_triggered = false;
    let mut timeout_triggered = false;
    let start_time = Instant::now();

    let _generated = engine.generate_stream(
        &input_ids,
        device,
        gen_cfg,
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
            if let Some(text) = utf8_buffer.push_and_decode(token_id, tokenizer)? {
                let filtered = think_filter.process(&text);
                if !filtered.is_empty() {
                    assistant_buf.push_str(&filtered);
                }
            }
            Ok(true)
        },
    )?;

    if let Some(tail_text) = utf8_buffer.flush(tokenizer)? {
        let filtered = think_filter.process(&tail_text);
        if !filtered.is_empty() {
            assistant_buf.push_str(&filtered);
        }
    }
    let tail = think_filter.flush();
    if !tail.is_empty() {
        assistant_buf.push_str(&tail);
    }

    if timeout_triggered {
        if !assistant_buf.trim().is_empty() {
            return Ok(SingleResponseResult {
                text: assistant_buf,
                truncated_reason: Some("generation timed out"),
            });
        }
        return Err(E::msg("generation timed out in single-prompt mode"));
    }
    if loop_triggered {
        if !assistant_buf.trim().is_empty() {
            return Ok(SingleResponseResult {
                text: assistant_buf,
                truncated_reason: Some("repeated output detected"),
            });
        }
        return Err(E::msg("generation stopped after detecting repeated output"));
    }

    Ok(SingleResponseResult {
        text: assistant_buf,
        truncated_reason: None,
    })
}

struct SingleResponseResult {
    text: String,
    truncated_reason: Option<&'static str>,
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

fn append_disable_think_hint(base: Option<&str>) -> String {
    let suffix = "请直接给出最终答案，不输出思考过程，也不要输出<think>标签。";
    match base.map(|s| s.trim()).filter(|s| !s.is_empty()) {
        Some(s) => format!("{s}\n{suffix}"),
        None => suffix.to_string(),
    }
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

fn history_pairs(history: &[(String, String)]) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let mut idx = 0;
    while idx + 1 < history.len() {
        let (role_a, text_a) = &history[idx];
        let (role_b, text_b) = &history[idx + 1];
        if role_a == "user" && role_b == "assistant" {
            pairs.push((text_a.clone(), text_b.clone()));
        }
        idx += 2;
    }
    pairs
}

fn build_truncated_prompt(
    model_arch: ModelArch,
    pairs: &[(String, String)],
    user_text: &str,
    system_prompt: Option<&str>,
    tokenizer: &Tokenizer,
    max_ctx: usize,
    max_new_tokens: usize,
) -> Result<(Vec<u32>, usize)> {
    let mut start = 0usize;
    loop {
        let kept = &pairs[start..];
        let prompt = render_history_prompt(model_arch, kept, user_text, system_prompt);
        let tokens = tokenizer.encode(prompt.clone(), false).map_err(E::msg)?;
        let input_ids = tokens.get_ids().to_vec();
        let expected_max = input_ids.len() + max_new_tokens + 8;
        if expected_max <= max_ctx || start >= pairs.len() {
            return Ok((input_ids, kept.len()));
        }
        start += 1;
    }
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

    fn push_and_decode(&mut self, token_id: u32, tokenizer: &Tokenizer) -> Result<Option<String>> {
        self.pending_ids.push(token_id);
        let text = tokenizer.decode(&self.pending_ids, true).map_err(E::msg)?;
        if text.contains('\u{FFFD}') {
            Ok(None)
        } else {
            self.pending_ids.clear();
            Ok(Some(text))
        }
    }

    fn flush(&mut self, tokenizer: &Tokenizer) -> Result<Option<String>> {
        if self.pending_ids.is_empty() {
            return Ok(None);
        }
        let text = tokenizer.decode(&self.pending_ids, true).map_err(E::msg)?;
        self.pending_ids.clear();
        Ok(Some(text))
    }
}

struct ThinkFilter {
    in_think: bool,
    pending: String,
}

impl ThinkFilter {
    fn new() -> Self {
        Self {
            in_think: false,
            pending: String::new(),
        }
    }

    fn process(&mut self, chunk: &str) -> String {
        let mut out = String::new();
        let mut buf = String::new();
        buf.push_str(&self.pending);
        buf.push_str(chunk);
        self.pending.clear();

        loop {
            if self.in_think {
                if let Some(idx) = buf.find("</think>") {
                    buf.drain(..idx + "</think>".len());
                    self.in_think = false;
                } else {
                    let keep = partial_suffix_len(&buf, "</think>");
                    if keep > 0 {
                        self.pending = buf[buf.len() - keep..].to_string();
                    }
                    break;
                }
            } else if let Some(idx) = buf.find("<think>") {
                out.push_str(&buf[..idx]);
                buf.drain(..idx + "<think>".len());
                self.in_think = true;
            } else {
                let keep = partial_suffix_len(&buf, "<think>");
                if keep > 0 {
                    let cut = buf.len() - keep;
                    out.push_str(&buf[..cut]);
                    self.pending = buf[cut..].to_string();
                } else {
                    out.push_str(&buf);
                }
                break;
            }
        }

        out
    }

    fn flush(&mut self) -> String {
        if self.in_think {
            self.pending.clear();
            return String::new();
        }
        std::mem::take(&mut self.pending)
    }
}

fn partial_suffix_len(s: &str, tag: &str) -> usize {
    let max = tag.len().saturating_sub(1).min(s.len());
    for len in (1..=max).rev() {
        if s.ends_with(&tag[..len]) {
            return len;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_model_id_prefers_cli_then_env_then_config() {
        let (model, source) = resolve_model_id(
            Some("cli-model".to_string()),
            Some("env-model".to_string()),
            Some("cfg-model".to_string()),
            "default-model",
        );
        assert_eq!(model, "cli-model");
        assert_eq!(source, "source=cli");

        let (model, source) = resolve_model_id(
            None,
            Some("env-model".to_string()),
            Some("cfg-model".to_string()),
            "default-model",
        );
        assert_eq!(model, "env-model");
        assert_eq!(source, "source=env");
    }

    #[test]
    fn think_filter_strips_think_blocks() {
        let mut filter = ThinkFilter::new();
        let first = filter.process("hello<think>internal");
        let second = filter.process(" note</think>world");
        let tail = filter.flush();
        assert_eq!(first, "hello");
        assert_eq!(second, "world");
        assert_eq!(tail, "");
    }
}
