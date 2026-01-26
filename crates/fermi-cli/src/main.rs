use anyhow::{Error as E, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use fermi_io::{download_qwen3_files, load_qwen3_config, load_tokenizer};
use fermi_runtime::{GenerationConfig, Qwen3Engine};
use tokenizers::Tokenizer;
use std::env;
use std::io::{self, Write};
use std::time::Instant;

fn main() -> Result<()> {
    let cli_cfg = parse_args()?;
    // 1. 基础环境设置
    let device = device_setup()?;
    println!("🚀 运行设备: {:?}", device);

    // ==========================================
    // 指定 Qwen3 官方模型 ID / 本地路径
    // ==========================================
    let model_repo_id = cli_cfg
        .model
        .clone()
        .or_else(|| std::env::var("FERMI_MODEL").ok())
        .unwrap_or_else(|| "Qwen/Qwen3-1.7B".to_string());
    
    println!("📥 准备模型文件...");
    println!("📦 模型: {}", model_repo_id);
    let files = download_qwen3_files(&model_repo_id, !cli_cfg.offline)?;
    println!("📥 检测到模型为分片格式，开始下载权重...");
    println!("✅ 权重下载完成");

    println!("⚙️ 正在解析配置文件...");
    let config = load_qwen3_config(&files.config)?;
    if config.sliding_window.is_some() {
        println!("⚠️  Config 修复: 将 'sliding_window' 设为 {}", config.max_position_embeddings);
    }

    // 5. 加载权重
    let dtype = if device.is_metal() { DType::F16 } else { DType::F32 };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files.weights, dtype, &device)? };

    // 6. 初始化模型 (使用自定义的 Qwen3Model)
    println!("🏗️ 正在构建模型架构 (Custom Qwen3 No-Bias)...");
    let mut engine = Qwen3Engine::new(&config, vb)?;
    let tokenizer = load_tokenizer(&files.tokenizer)?;
    engine.clear_kv_cache();

    println!("💬 交互模式：输入问题，回车发送。命令：/help /reset /exit");

    let mut history: Vec<(String, String)> = Vec::new();
    let mut current_pos: usize = 0;
    let mut has_context = false;
    let im_end_id = tokenizer.token_to_id("<|im_end|>");
    let mut stop_tokens = Vec::new();
    if let Some(eos) = tokenizer.token_to_id("<|endoftext|>") {
        stop_tokens.push(eos);
    }
    if let Some(im_end) = tokenizer.token_to_id("<|im_end|>") {
        stop_tokens.push(im_end);
    }

    let gen_cfg = GenerationConfig {
        max_new_tokens: cli_cfg.max_new_tokens,
        repeat_penalty: cli_cfg.repeat_penalty,
        stop_tokens,
        temperature: cli_cfg.temperature,
        top_p: cli_cfg.top_p,
    };
    let max_ctx = config.max_position_embeddings;
    let timeout_ms = cli_cfg.timeout_ms;

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
            .encode(render_user_chunk(line, has_context), false)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let expected_max = offset + input_ids.len() + gen_cfg.max_new_tokens + 8;
        if expected_max > max_ctx {
            let pairs = history_pairs(&history);
            let (trunc_ids, kept_pairs) =
                build_truncated_prompt(&pairs, line, &tokenizer, max_ctx, gen_cfg.max_new_tokens)?;
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

        let generated = engine.generate_stream_with_offset(
            &input_ids,
            offset,
            &device,
            &gen_cfg,
            |token_id| {
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
                if let Some(im_end) = im_end_id {
                    if last_token != im_end {
                        engine.append_tokens(&[im_end], cache_len, &device)?;
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
    max_new_tokens: usize,
    repeat_penalty: f32,
    temperature: f32,
    top_p: f32,
    model: Option<String>,
    offline: bool,
    timeout_ms: u64,
}

fn parse_args() -> Result<CliConfig> {
    let mut max_new_tokens = 1024usize;
    let mut repeat_penalty = 1.1f32;
    let mut temperature = 0.8f32;
    let mut top_p = 0.95f32;
    let mut model: Option<String> = None;
    let mut offline = false;
    let mut timeout_ms: u64 = 60_000;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => {
                if let Some(v) = args.next() {
                    model = Some(v);
                } else {
                    return Err(E::msg("--model requires a value"));
                }
            }
            "--offline" => {
                offline = true;
            }
            "--timeout-ms" => {
                if let Some(v) = args.next() {
                    timeout_ms = v.parse::<u64>().map_err(E::msg)?;
                } else {
                    return Err(E::msg("--timeout-ms requires a value"));
                }
            }
            "--max-new-tokens" => {
                if let Some(v) = args.next() {
                    max_new_tokens = v.parse::<usize>().map_err(E::msg)?;
                } else {
                    return Err(E::msg("--max-new-tokens requires a value"));
                }
            }
            "--repeat-penalty" => {
                if let Some(v) = args.next() {
                    repeat_penalty = v.parse::<f32>().map_err(E::msg)?;
                } else {
                    return Err(E::msg("--repeat-penalty requires a value"));
                }
            }
            "--temperature" => {
                if let Some(v) = args.next() {
                    temperature = v.parse::<f32>().map_err(E::msg)?;
                } else {
                    return Err(E::msg("--temperature requires a value"));
                }
            }
            "--top-p" => {
                if let Some(v) = args.next() {
                    top_p = v.parse::<f32>().map_err(E::msg)?;
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
        timeout_ms,
    })
}

fn print_usage() {
    println!("Usage: fermi-infer [--model ID|PATH] [--offline] [--timeout-ms MS] [--max-new-tokens N] [--repeat-penalty P] [--temperature T] [--top-p P]");
    println!("  --model           HuggingFace repo id or local model dir (default: Qwen/Qwen3-1.7B)");
    println!("  --offline         Disable network access; require local model files");
    println!("  --timeout-ms      Per-request timeout in milliseconds (default: 60000; 0 disables)");
    println!("  --max-new-tokens  Maximum number of generated tokens (default: 1024)");
    println!("  --repeat-penalty  Repetition penalty (default: 1.1)");
    println!("  --temperature     Sampling temperature (default: 0.8)");
    println!("  --top-p           Nucleus sampling p (default: 0.95)");
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

fn render_history_prompt(pairs: &[(String, String)], user_text: &str) -> String {
    let mut out = String::new();
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

fn build_truncated_prompt(
    pairs: &[(String, String)],
    user_text: &str,
    tokenizer: &Tokenizer,
    max_ctx: usize,
    max_new_tokens: usize,
) -> Result<(Vec<u32>, usize)> {
    let mut start = 0usize;
    loop {
        let kept = &pairs[start..];
        let prompt = render_history_prompt(kept, user_text);
        let tokens = tokenizer.encode(prompt.clone(), false).map_err(E::msg)?;
        let input_ids = tokens.get_ids().to_vec();
        let expected_max = input_ids.len() + max_new_tokens + 8;
        if expected_max <= max_ctx || start >= pairs.len() {
            return Ok((input_ids, kept.len()));
        }
        start += 1;
    }
}

fn render_user_chunk(user_text: &str, has_context: bool) -> String {
    let mut out = String::new();
    if has_context {
        out.push('\n');
    }
    out.push_str("<|im_start|>user\n");
    out.push_str(user_text);
    out.push_str("<|im_end|>\n<|im_start|>assistant\n");
    out
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

    fn push_and_decode(
        &mut self,
        token_id: u32,
        tokenizer: &Tokenizer,
    ) -> Result<Option<String>> {
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
