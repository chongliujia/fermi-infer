use anyhow::{Error as E, Result};
use candle_core::Device;
use fermi_io::load_tokenizer;
use fermi_runtime::{
    GenerationConfig, ModelBuilder, SamplingDefaults, apply_sampling_preset, build_stop_tokens,
    load_config, parse_sampling_preset, render_chat_prompt, resolve_sampling_params,
    sampling_defaults_from_sources,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

const DEFAULT_MODEL: &str = "Qwen/Qwen3-1.7B";

fn main() -> Result<()> {
    let cli = parse_args()?;
    let loaded_cfg = load_config(cli.config.as_deref())?;
    let app_cfg = loaded_cfg.config.clone();

    let device = device_setup()?;
    let model_id = cli
        .model
        .clone()
        .or_else(|| env::var("FERMI_MODEL").ok())
        .or_else(|| app_cfg.model.id.clone())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let offline = cli
        .offline
        .or_else(|| env_flag_opt("FERMI_OFFLINE"))
        .or_else(|| env_flag_opt("HF_HUB_OFFLINE"))
        .or(app_cfg.model.offline)
        .unwrap_or(false);

    let builder = ModelBuilder::new(&model_id, !offline)?;
    let mut sampling_defaults =
        sampling_defaults_from_sources(app_cfg.generation.to_sampling_overrides())?;
    if let Some(preset) = cli.preset.as_deref() {
        sampling_defaults =
            apply_sampling_preset(&sampling_defaults, builder.model_arch(), parse_sampling_preset(preset)?);
    }
    let sampling = resolve_sampling_params(
        cli.max_new_tokens,
        cli.temperature,
        cli.top_p,
        cli.repeat_penalty,
        &sampling_defaults,
    )?;

    let mut engine = builder.create_engine(&device)?;
    let tokenizer = load_tokenizer(builder.tokenizer_path())?;
    let model_arch = builder.model_arch();
    let cases = load_cases(&cli)?;

    let gen_cfg = GenerationConfig {
        max_new_tokens: sampling.max_new_tokens,
        repeat_penalty: sampling.repeat_penalty,
        stop_tokens: build_stop_tokens(model_arch, &tokenizer),
        temperature: sampling.temperature,
        top_p: sampling.top_p,
    };

    let mut reports = Vec::with_capacity(cases.len());
    for case in cases {
        for _ in 0..cli.warmup {
            let _ = run_case(&mut *engine, &tokenizer, &device, model_arch, &case, &gen_cfg)?;
        }
        let report = run_case(&mut *engine, &tokenizer, &device, model_arch, &case, &gen_cfg)?;
        if !cli.json {
            print_case_report(&report);
        }
        reports.push(report);
    }

    let summary = EvalSummary::from_reports(
        model_id,
        cli.suite.clone(),
        cli.case_file.clone(),
        cli.warmup,
        reports,
        &sampling_defaults,
        &gen_cfg,
    );
    let output_json = serde_json::to_string_pretty(&summary)?;

    if let Some(path) = &cli.out {
        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, &output_json)?;
    }

    if cli.json {
        println!("{}", output_json);
    } else {
        print_summary(&summary);
        if let Some(path) = &cli.out {
            println!("report_written: {}", path);
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct CliConfig {
    model: Option<String>,
    offline: Option<bool>,
    config: Option<String>,
    suite: Option<String>,
    case_file: Option<String>,
    max_new_tokens: Option<usize>,
    repeat_penalty: Option<f32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    json: bool,
    out: Option<String>,
    warmup: usize,
    preset: Option<String>,
}

fn parse_args() -> Result<CliConfig> {
    let mut cli = CliConfig {
        model: None,
        offline: None,
        config: None,
        suite: Some("en-basic".to_string()),
        case_file: None,
        max_new_tokens: None,
        repeat_penalty: None,
        temperature: None,
        top_p: None,
        json: false,
        out: None,
        warmup: 0,
        preset: None,
    };

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => cli.model = Some(next_value(&mut args, "--model")?),
            "--offline" => cli.offline = Some(true),
            "--online" => cli.offline = Some(false),
            "--config" => cli.config = Some(next_value(&mut args, "--config")?),
            "--suite" => cli.suite = Some(next_value(&mut args, "--suite")?),
            "--case-file" => cli.case_file = Some(next_value(&mut args, "--case-file")?),
            "--max-new-tokens" => {
                cli.max_new_tokens = Some(
                    next_value(&mut args, "--max-new-tokens")?
                        .parse::<usize>()
                        .map_err(E::msg)?,
                )
            }
            "--repeat-penalty" => {
                cli.repeat_penalty = Some(
                    next_value(&mut args, "--repeat-penalty")?
                        .parse::<f32>()
                        .map_err(E::msg)?,
                )
            }
            "--temperature" => {
                cli.temperature = Some(
                    next_value(&mut args, "--temperature")?
                        .parse::<f32>()
                        .map_err(E::msg)?,
                )
            }
            "--top-p" => {
                cli.top_p = Some(
                    next_value(&mut args, "--top-p")?
                        .parse::<f32>()
                        .map_err(E::msg)?,
                )
            }
            "--warmup" => {
                cli.warmup = next_value(&mut args, "--warmup")?
                    .parse::<usize>()
                    .map_err(E::msg)?
            }
            "--preset" => cli.preset = Some(next_value(&mut args, "--preset")?),
            "--json" => cli.json = true,
            "--out" => cli.out = Some(next_value(&mut args, "--out")?),
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(E::msg(format!("unknown argument: {}", other))),
        }
    }

    if cli.case_file.is_some() {
        cli.suite = None;
    }

    Ok(cli)
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| E::msg(format!("{} requires a value", flag)))
}

fn print_usage() {
    println!(
        "Usage: fermi-eval [--config PATH] [--model ID|PATH] [--suite NAME|--case-file PATH] [--preset NAME] [--offline|--online] [--warmup N] [--max-new-tokens N] [--repeat-penalty P] [--temperature T] [--top-p P] [--json] [--out PATH]"
    );
    println!("  --suite           Built-in evaluation suite (default: en-basic)");
    println!("  --case-file       Load eval cases from a JSON file");
    println!("  --preset          Sampling preset: chat-balanced, chat-precise, reasoning, creative");
    println!("  --out             Write evaluation report to a JSON file");
}

fn device_setup() -> Result<Device> {
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

#[derive(Debug, Clone, Deserialize)]
struct EvalCase {
    name: String,
    category: String,
    prompt: String,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    required_all: Vec<String>,
    #[serde(default)]
    required_any: Vec<String>,
    #[serde(default)]
    forbidden_any: Vec<String>,
    #[serde(default)]
    min_chars: Option<usize>,
    #[serde(default)]
    max_chars: Option<usize>,
    #[serde(default)]
    exact_sentence_count: Option<usize>,
}

fn load_cases(cli: &CliConfig) -> Result<Vec<EvalCase>> {
    if let Some(path) = &cli.case_file {
        let text = fs::read_to_string(path)?;
        return serde_json::from_str(&text).map_err(E::msg);
    }

    match cli.suite.as_deref().unwrap_or("en-basic") {
        "en-basic" => Ok(default_en_basic_suite()),
        "cn-basic" => Ok(default_cn_basic_suite()),
        other => Err(E::msg(format!(
            "unsupported suite '{}', supported values: en-basic, cn-basic",
            other
        ))),
    }
}

fn default_en_basic_suite() -> Vec<EvalCase> {
    vec![
        EvalCase {
            name: "first_principles".to_string(),
            category: "chat".to_string(),
            prompt: "Explain what first-principles thinking is and give one engineering example."
                .to_string(),
            system_prompt: Some("You are a rigorous technical assistant.".to_string()),
            required_all: vec!["first-principles".to_string()],
            required_any: vec![
                "fundamental".to_string(),
                "assumption".to_string(),
                "break".to_string(),
            ],
            forbidden_any: vec!["As an AI".to_string()],
            min_chars: Some(40),
            max_chars: None,
            exact_sentence_count: None,
        },
        EvalCase {
            name: "photosynthesis".to_string(),
            category: "knowledge".to_string(),
            prompt: "Briefly explain what photosynthesis does and mention what it consumes and what it produces."
                .to_string(),
            system_prompt: None,
            required_all: vec!["photosynthesis".to_string()],
            required_any: vec![
                "carbon dioxide".to_string(),
                "oxygen".to_string(),
                "glucose".to_string(),
                "sugar".to_string(),
                "sunlight".to_string(),
            ],
            forbidden_any: vec!["As an AI".to_string()],
            min_chars: Some(30),
            max_chars: None,
            exact_sentence_count: None,
        },
        EvalCase {
            name: "python_fib".to_string(),
            category: "code".to_string(),
            prompt: "Write a Python function fib(n) that returns the nth Fibonacci number, and include a short example call."
                .to_string(),
            system_prompt: Some("Return code first, then one short explanation.".to_string()),
            required_all: vec![],
            required_any: vec![
                "def fib".to_string(),
                "def fibonacci".to_string(),
                "fib(".to_string(),
            ],
            forbidden_any: vec!["As an AI".to_string()],
            min_chars: Some(20),
            max_chars: None,
            exact_sentence_count: None,
        },
        EvalCase {
            name: "reasoning_scale".to_string(),
            category: "reasoning".to_string(),
            prompt: "There are 8 balls and 1 has a different weight, but you do not know whether it is heavier or lighter. With only two weighings, can you always identify the odd ball and whether it is heavier or lighter? Give the conclusion."
                .to_string(),
            system_prompt: Some("Give the conclusion directly and justify it briefly.".to_string()),
            required_any: vec!["no".to_string(), "cannot".to_string(), "not always".to_string()],
            required_all: vec![],
            forbidden_any: vec!["As an AI".to_string()],
            min_chars: Some(10),
            max_chars: None,
            exact_sentence_count: None,
        },
        EvalCase {
            name: "system_following".to_string(),
            category: "instruction".to_string(),
            prompt: "Introduce Rust.".to_string(),
            system_prompt: Some("Answer in exactly two sentences and mention memory safety.".to_string()),
            required_all: vec!["memory safety".to_string()],
            required_any: vec![],
            forbidden_any: vec![],
            min_chars: Some(10),
            max_chars: Some(220),
            exact_sentence_count: Some(2),
        },
        EvalCase {
            name: "summary_arch".to_string(),
            category: "summary".to_string(),
            prompt: "An inference system uses a layered design: the entry layer handles protocol translation and request governance, the runtime layer handles scheduling, sampling, and session management, the model layer handles attention, KV cache, and forward computation, and the I/O layer handles model download, caching, and config compatibility. Summarize the architecture."
                .to_string(),
            system_prompt: None,
            required_any: vec![
                "layer".to_string(),
                "scheduling".to_string(),
                "KV".to_string(),
            ],
            required_all: vec![],
            forbidden_any: vec!["As an AI".to_string()],
            min_chars: Some(30),
            max_chars: None,
            exact_sentence_count: None,
        },
    ]
}

fn default_cn_basic_suite() -> Vec<EvalCase> {
    vec![
        EvalCase {
            name: "first_principles".to_string(),
            category: "chat".to_string(),
            prompt: "请用中文解释什么是第一性原理，并给一个工程例子。".to_string(),
            system_prompt: Some("你是一个严谨的中文技术助手。".to_string()),
            required_all: vec!["第一性原理".to_string()],
            required_any: vec!["本质".to_string(), "基本假设".to_string(), "拆解".to_string()],
            forbidden_any: vec!["作为一个AI".to_string()],
            min_chars: Some(40),
            max_chars: None,
            exact_sentence_count: None,
        },
        EvalCase {
            name: "photosynthesis".to_string(),
            category: "knowledge".to_string(),
            prompt: "请用中文简要说明光合作用的作用，并提到会消耗什么、产生什么。".to_string(),
            system_prompt: None,
            required_all: vec!["光合作用".to_string()],
            required_any: vec![
                "二氧化碳".to_string(),
                "氧气".to_string(),
                "葡萄糖".to_string(),
                "糖".to_string(),
                "阳光".to_string(),
            ],
            forbidden_any: vec!["作为一个AI".to_string()],
            min_chars: Some(30),
            max_chars: None,
            exact_sentence_count: None,
        },
        EvalCase {
            name: "python_fib".to_string(),
            category: "code".to_string(),
            prompt: "请写一个 Python 函数 fib(n)，返回斐波那契数列第 n 项，并给一个简单调用示例。".to_string(),
            system_prompt: Some("请直接给出代码和简短说明。".to_string()),
            required_all: vec![],
            required_any: vec![
                "def fib".to_string(),
                "def fibonacci".to_string(),
                "fib(".to_string(),
            ],
            forbidden_any: vec!["作为一个AI".to_string()],
            min_chars: Some(20),
            max_chars: None,
            exact_sentence_count: None,
        },
        EvalCase {
            name: "reasoning_scale".to_string(),
            category: "reasoning".to_string(),
            prompt: "有 8 个球，其中 1 个重量不同，但不知道更重还是更轻。只称 2 次，是否一定能找出异常球并判断轻重？请给结论。".to_string(),
            system_prompt: Some("请直接给结论，并用一两句话说明理由。".to_string()),
            required_any: vec!["不能".to_string(), "不一定".to_string(), "无法".to_string()],
            required_all: vec![],
            forbidden_any: vec!["作为一个AI".to_string()],
            min_chars: Some(10),
            max_chars: None,
            exact_sentence_count: None,
        },
        EvalCase {
            name: "system_following".to_string(),
            category: "instruction".to_string(),
            prompt: "请介绍一下 Rust。".to_string(),
            system_prompt: Some("只用两句话回答，并且必须提到内存安全。".to_string()),
            required_all: vec!["内存安全".to_string()],
            required_any: vec![],
            forbidden_any: vec![],
            min_chars: Some(10),
            max_chars: Some(180),
            exact_sentence_count: Some(2),
        },
        EvalCase {
            name: "summary_arch".to_string(),
            category: "summary".to_string(),
            prompt: "某推理系统采用分层设计：入口层负责协议转换和请求治理，运行时层负责调度、采样与会话管理，模型层负责 attention、KV cache 与前向计算，I/O 层负责模型下载、缓存与配置兼容。请用中文总结其架构要点。".to_string(),
            system_prompt: None,
            required_any: vec!["分层".to_string(), "调度".to_string(), "KV".to_string()],
            required_all: vec![],
            forbidden_any: vec!["作为一个AI".to_string()],
            min_chars: Some(30),
            max_chars: None,
            exact_sentence_count: None,
        },
    ]
}

#[derive(Debug, Clone, Serialize)]
struct EvalCaseReport {
    name: String,
    category: String,
    passed: bool,
    score: f64,
    failure_reasons: Vec<String>,
    prompt_tokens: usize,
    completion_tokens: usize,
    ttft_ms: f64,
    total_ms: f64,
    tokens_per_second: f64,
    output_chars: usize,
    output: String,
}

fn run_case(
    engine: &mut dyn fermi_runtime::InferenceEngine,
    tokenizer: &Tokenizer,
    device: &Device,
    model_arch: fermi_io::ModelArch,
    case: &EvalCase,
    cfg: &GenerationConfig,
) -> Result<EvalCaseReport> {
    engine.clear_kv_cache();
    let prompt = render_chat_prompt(model_arch, &case.prompt, case.system_prompt.as_deref());
    let input_ids = tokenizer.encode(prompt, false).map_err(E::msg)?.get_ids().to_vec();
    let prompt_tokens = input_ids.len();
    let start = Instant::now();
    let mut ttft = None;
    let mut utf8_buffer = Utf8Buffer::new();
    let mut output = String::new();

    let generated = engine.generate_stream(&input_ids, device, cfg, &mut |token_id| {
        if ttft.is_none() {
            ttft = Some(start.elapsed());
        }
        if let Some(text) = utf8_buffer.push_and_decode(token_id, tokenizer)? {
            output.push_str(&text);
        }
        Ok(true)
    })?;
    if let Some(tail) = utf8_buffer.flush(tokenizer)? {
        output.push_str(&tail);
    }
    let total = start.elapsed();
    let completion_tokens = generated.len();
    let tokens_per_second = if total.as_secs_f64() > 0.0 {
        completion_tokens as f64 / total.as_secs_f64()
    } else {
        0.0
    };

    let failure_reasons = evaluate_output(case, &output);
    let passed = failure_reasons.is_empty();
    let score = compute_score(&failure_reasons);

    Ok(EvalCaseReport {
        name: case.name.clone(),
        category: case.category.clone(),
        passed,
        score,
        failure_reasons,
        prompt_tokens,
        completion_tokens,
        ttft_ms: duration_ms(ttft.unwrap_or(total)),
        total_ms: duration_ms(total),
        tokens_per_second,
        output_chars: output.chars().count(),
        output: output.trim().to_string(),
    })
}

fn evaluate_output(case: &EvalCase, output: &str) -> Vec<String> {
    let mut failures = Vec::new();
    let trimmed = output.trim();
    let lowered = trimmed.to_ascii_lowercase();

    if trimmed.is_empty() {
        failures.push("empty output".to_string());
        return failures;
    }

    if let Some(min_chars) = case.min_chars {
        if trimmed.chars().count() < min_chars {
            failures.push(format!("output shorter than min_chars={}", min_chars));
        }
    }
    if let Some(max_chars) = case.max_chars {
        if trimmed.chars().count() > max_chars {
            failures.push(format!("output longer than max_chars={}", max_chars));
        }
    }
    if let Some(exact_sentence_count) = case.exact_sentence_count {
        let sentence_count = count_sentences(trimmed);
        if sentence_count != exact_sentence_count {
            failures.push(format!(
                "sentence count {} != expected {}",
                sentence_count, exact_sentence_count
            ));
        }
    }

    for needle in &case.required_all {
        if !trimmed.contains(needle) {
            failures.push(format!("missing required text '{}'", needle));
        }
    }

    if !case.required_any.is_empty()
        && !case.required_any.iter().any(|needle| trimmed.contains(needle))
    {
        failures.push(format!(
            "missing any required text from {:?}",
            case.required_any
        ));
    }

    for needle in &case.forbidden_any {
        if trimmed.contains(needle) {
            failures.push(format!("contains forbidden text '{}'", needle));
        }
    }

    if has_repeated_suffix(trimmed) {
        failures.push("repeated trailing pattern detected".to_string());
    }
    if lowered.contains("<think>") || lowered.contains("</think>") {
        failures.push("raw think tags leaked".to_string());
    }

    failures
}

fn has_repeated_suffix(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    for width in 2..=12 {
        if chars.len() < width * 3 {
            continue;
        }
        let a = &chars[chars.len() - width..];
        let b = &chars[chars.len() - 2 * width..chars.len() - width];
        let c = &chars[chars.len() - 3 * width..chars.len() - 2 * width];
        if a == b && b == c {
            return true;
        }
    }
    false
}

fn count_sentences(text: &str) -> usize {
    let mut count = 0usize;
    let mut in_text = false;
    for ch in text.chars() {
        if !ch.is_whitespace() {
            in_text = true;
        }
        if matches!(ch, '.' | '!' | '?' | '。' | '！' | '？') && in_text {
            count += 1;
            in_text = false;
        }
    }
    if in_text {
        count += 1;
    }
    count
}

fn compute_score(failures: &[String]) -> f64 {
    if failures.is_empty() {
        1.0
    } else {
        (1.0 - 0.2 * failures.len() as f64).max(0.0)
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[derive(Debug, Serialize)]
struct EvalSummary {
    model: String,
    suite: Option<String>,
    case_file: Option<String>,
    warmup_runs: usize,
    total_cases: usize,
    passed_cases: usize,
    failed_cases: usize,
    pass_rate: f64,
    avg_score: f64,
    avg_ttft_ms: f64,
    avg_total_ms: f64,
    avg_tokens_per_second: f64,
    max_new_tokens: usize,
    temperature: f32,
    top_p: f32,
    repeat_penalty: f32,
    reports: Vec<EvalCaseReport>,
}

impl EvalSummary {
    fn from_reports(
        model: String,
        suite: Option<String>,
        case_file: Option<String>,
        warmup_runs: usize,
        reports: Vec<EvalCaseReport>,
        _defaults: &SamplingDefaults,
        cfg: &GenerationConfig,
    ) -> Self {
        let total_cases = reports.len();
        let passed_cases = reports.iter().filter(|r| r.passed).count();
        let failed_cases = total_cases.saturating_sub(passed_cases);
        let pass_rate = if total_cases > 0 {
            passed_cases as f64 / total_cases as f64
        } else {
            0.0
        };

        Self {
            model,
            suite,
            case_file,
            warmup_runs,
            total_cases,
            passed_cases,
            failed_cases,
            pass_rate,
            avg_score: average(reports.iter().map(|r| r.score)),
            avg_ttft_ms: average(reports.iter().map(|r| r.ttft_ms)),
            avg_total_ms: average(reports.iter().map(|r| r.total_ms)),
            avg_tokens_per_second: average(reports.iter().map(|r| r.tokens_per_second)),
            max_new_tokens: cfg.max_new_tokens,
            temperature: cfg.temperature,
            top_p: cfg.top_p,
            repeat_penalty: cfg.repeat_penalty,
            reports,
        }
    }
}

fn average(values: impl Iterator<Item = f64>) -> f64 {
    let mut total = 0.0;
    let mut count = 0u64;
    for value in values {
        total += value;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn print_case_report(report: &EvalCaseReport) {
    let status = if report.passed { "PASS" } else { "FAIL" };
    println!(
        "[{}] {} ({}) score={:.2} ttft={:.2} ms total={:.2} ms tok/s={:.2}",
        status,
        report.name,
        report.category,
        report.score,
        report.ttft_ms,
        report.total_ms,
        report.tokens_per_second
    );
    if !report.failure_reasons.is_empty() {
        println!("  reasons: {}", report.failure_reasons.join("; "));
    }
}

fn print_summary(summary: &EvalSummary) {
    println!();
    println!("model: {}", summary.model);
    if let Some(suite) = &summary.suite {
        println!("suite: {}", suite);
    }
    if let Some(case_file) = &summary.case_file {
        println!("case_file: {}", case_file);
    }
    println!(
        "summary: passed={}/{} failed={} pass_rate={:.2} avg_score={:.2}",
        summary.passed_cases,
        summary.total_cases,
        summary.failed_cases,
        summary.pass_rate,
        summary.avg_score
    );
    println!(
        "performance: avg_ttft_ms={:.2} avg_total_ms={:.2} avg_tokens_per_second={:.2}",
        summary.avg_ttft_ms,
        summary.avg_total_ms,
        summary.avg_tokens_per_second
    );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_output_flags_missing_required_terms() {
        let case = EvalCase {
            name: "x".to_string(),
            category: "x".to_string(),
            prompt: "x".to_string(),
            system_prompt: None,
            required_all: vec!["第一性原理".to_string()],
            required_any: vec!["本质".to_string()],
            forbidden_any: vec![],
            min_chars: Some(5),
            max_chars: None,
            exact_sentence_count: None,
        };
        let failures = evaluate_output(&case, "太短");
        assert!(failures.iter().any(|msg| msg.contains("required text")));
    }

    #[test]
    fn repeated_suffix_detection_catches_loops() {
        assert!(has_repeated_suffix("abcxyzxyzxyz"));
        assert!(!has_repeated_suffix("正常回答，没有明显重复"));
    }

    #[test]
    fn counts_sentences_from_terminal_punctuation() {
        assert_eq!(count_sentences("Rust is fast. Rust is safe."), 2);
        assert_eq!(count_sentences("一句话。两句话。"), 2);
    }
}
