use anyhow::{Error as E, Result};
use candle_core::Device;
use fermi_io::{ModelArch, load_tokenizer};
use fermi_runtime::{
    GenerationConfig, ModelBuilder, SamplingDefaults, apply_sampling_preset, build_stop_tokens,
    load_config, parse_sampling_preset, render_chat_prompt, resolve_sampling_params,
    sampling_defaults_from_sources,
};
use serde::Serialize;
use std::env;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

const DEFAULT_MODEL: &str = "Qwen/Qwen3-1.7B";
const DEFAULT_PROMPT: &str = "Explain what first-principles thinking is.";

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
    let max_ctx = builder.max_position_embeddings();
    let model_arch = builder.model_arch();

    let gen_cfg = GenerationConfig {
        max_new_tokens: sampling.max_new_tokens,
        repeat_penalty: sampling.repeat_penalty,
        stop_tokens: build_stop_tokens(model_arch, &tokenizer),
        temperature: sampling.temperature,
        top_p: sampling.top_p,
    };

    let output_json = if let Some(suite_name) = cli.suite.as_deref() {
        let cases = build_suite_cases(suite_name, model_arch)?;
        let mut reports = Vec::with_capacity(cases.len());
        for case in cases {
            let report = run_benchmark(
                &mut *engine,
                &tokenizer,
                &device,
                max_ctx,
                &model_id,
                &sampling_defaults,
                &gen_cfg,
                &cli,
                case.name,
                case.prompt,
            )?;
            if !cli.json {
                print_report(&report);
                println!();
            }
            reports.push(report);
        }
        serde_json::to_string_pretty(&SuiteReport {
            model: model_id.clone(),
            warmup_runs: cli.warmup,
            measured_runs: cli.runs,
            reports,
        })?
    } else {
        let prompt_text = load_prompt(&cli)?;
        let prompt = if cli.raw_prompt {
            prompt_text
        } else {
            render_chat_prompt(model_arch, prompt_text.as_str(), cli.system_prompt.as_deref())
        };
        let report = run_benchmark(
            &mut *engine,
            &tokenizer,
            &device,
            max_ctx,
            &model_id,
            &sampling_defaults,
            &gen_cfg,
            &cli,
            "custom",
            prompt,
        )?;
        if !cli.json {
            print_report(&report);
        }
        serde_json::to_string_pretty(&report)?
    };

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
    } else if let Some(path) = &cli.out {
        println!("report_written: {}", path);
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct CliConfig {
    model: Option<String>,
    offline: Option<bool>,
    config: Option<String>,
    prompt: Option<String>,
    prompt_file: Option<String>,
    system_prompt: Option<String>,
    raw_prompt: bool,
    runs: usize,
    warmup: usize,
    max_new_tokens: Option<usize>,
    repeat_penalty: Option<f32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    json: bool,
    out: Option<String>,
    suite: Option<String>,
    preset: Option<String>,
}

fn parse_args() -> Result<CliConfig> {
    let mut cli = CliConfig {
        model: None,
        offline: None,
        config: None,
        prompt: None,
        prompt_file: None,
        system_prompt: None,
        raw_prompt: false,
        runs: 5,
        warmup: 1,
        max_new_tokens: None,
        repeat_penalty: None,
        temperature: None,
        top_p: None,
        json: false,
        out: None,
        suite: None,
        preset: None,
    };

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => cli.model = Some(next_value(&mut args, "--model")?),
            "--offline" => cli.offline = Some(true),
            "--online" => cli.offline = Some(false),
            "--config" => cli.config = Some(next_value(&mut args, "--config")?),
            "--prompt" => cli.prompt = Some(next_value(&mut args, "--prompt")?),
            "--prompt-file" => cli.prompt_file = Some(next_value(&mut args, "--prompt-file")?),
            "--system-prompt" => {
                cli.system_prompt = Some(next_value(&mut args, "--system-prompt")?)
            }
            "--raw-prompt" => cli.raw_prompt = true,
            "--runs" => cli.runs = next_value(&mut args, "--runs")?.parse::<usize>().map_err(E::msg)?,
            "--warmup" => {
                cli.warmup = next_value(&mut args, "--warmup")?
                    .parse::<usize>()
                    .map_err(E::msg)?
            }
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
            "--json" => cli.json = true,
            "--out" => cli.out = Some(next_value(&mut args, "--out")?),
            "--suite" => cli.suite = Some(next_value(&mut args, "--suite")?),
            "--preset" => cli.preset = Some(next_value(&mut args, "--preset")?),
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(E::msg(format!("unknown argument: {}", other))),
        }
    }

    if cli.runs == 0 {
        return Err(E::msg("--runs must be > 0"));
    }
    if cli.prompt.is_some() && cli.prompt_file.is_some() {
        return Err(E::msg("use either --prompt or --prompt-file, not both"));
    }
    if cli.suite.is_some() && (cli.prompt.is_some() || cli.prompt_file.is_some()) {
        return Err(E::msg("use either --suite or --prompt/--prompt-file"));
    }

    Ok(cli)
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| E::msg(format!("{} requires a value", flag)))
}

fn print_usage() {
    println!(
        "Usage: fermi-bench [--config PATH] [--model ID|PATH] [--offline|--online] [--prompt TEXT|--prompt-file PATH] [--suite NAME] [--preset NAME] [--system-prompt TEXT] [--raw-prompt] [--runs N] [--warmup N] [--max-new-tokens N] [--repeat-penalty P] [--temperature T] [--top-p P] [--json] [--out PATH]"
    );
    println!("  --suite           Built-in benchmark suite: all");
    println!("  --preset          Sampling preset: chat-balanced, chat-precise, reasoning, creative");
    println!("  --out             Write benchmark report to a file in JSON format");
}

fn load_prompt(cli: &CliConfig) -> Result<String> {
    if let Some(prompt) = &cli.prompt {
        return Ok(prompt.clone());
    }
    if let Some(path) = &cli.prompt_file {
        return Ok(fs::read_to_string(path)?);
    }
    Ok(DEFAULT_PROMPT.to_string())
}

struct BenchmarkCase {
    name: &'static str,
    prompt: String,
}

fn build_suite_cases(name: &str, arch: ModelArch) -> Result<Vec<BenchmarkCase>> {
    match name {
        "all" => Ok(vec![
            BenchmarkCase {
                name: "short",
                prompt: render_chat_prompt(
                    arch,
                    "Explain what first-principles thinking is.",
                    Some("Give a concise and accurate answer."),
                ),
            },
            BenchmarkCase {
                name: "medium",
                prompt: render_chat_prompt(
                    arch,
                    &format!(
                        "{}\n\nSummarize the system design above and give five engineering recommendations.",
                        repeated_context(12)
                    ),
                    Some("You are a rigorous systems architecture assistant."),
                ),
            },
            BenchmarkCase {
                name: "long",
                prompt: render_chat_prompt(
                    arch,
                    &format!(
                        "{}\n\nSummarize the system from architecture, performance, and maintainability perspectives.",
                        repeated_context(40)
                    ),
                    Some("Answer using only the provided input."),
                ),
            },
            BenchmarkCase {
                name: "reasoning",
                prompt: render_chat_prompt(
                    arch,
                    "You have 8 balls and 1 has a different weight, but you do not know whether it is heavier or lighter. Can you always identify the odd ball and whether it is heavier or lighter using only two weighings? Analyze it.",
                    Some("Reason carefully and give a clear conclusion."),
                ),
            },
        ]),
        other => Err(E::msg(format!(
            "unsupported suite '{}', supported values: all",
            other
        ))),
    }
}

fn repeated_context(paragraphs: usize) -> String {
    let paragraph = "An inference system uses a layered design: the entry layer handles protocol translation and request governance, the runtime layer handles scheduling, sampling, and session management, the model layer handles attention, KV cache, and forward computation, and the I/O layer handles model download, caching, and config compatibility. The goal is to provide stable, low-latency, streaming inference on local devices while keeping module boundaries clear enough to extend to more model architectures and deployment forms.";
    let mut out = String::new();
    for idx in 0..paragraphs {
        if idx > 0 {
            out.push_str("\n\n");
        }
        out.push_str(paragraph);
    }
    out
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

#[derive(Debug, Clone, Serialize)]
struct RunResult {
    prompt_tokens: usize,
    completion_tokens: usize,
    ttft_ms: f64,
    total_ms: f64,
    tokens_per_second: f64,
}

fn run_once(
    engine: &mut dyn fermi_runtime::InferenceEngine,
    tokenizer: &tokenizers::Tokenizer,
    device: &Device,
    prompt: &str,
    cfg: &GenerationConfig,
) -> Result<RunResult> {
    engine.clear_kv_cache();
    let input_ids = tokenizer.encode(prompt, false).map_err(E::msg)?.get_ids().to_vec();
    let start = Instant::now();
    let mut ttft = None;
    let generated = engine.generate_stream(&input_ids, device, cfg, &mut |token_id| {
        if ttft.is_none() {
            ttft = Some(start.elapsed());
        }
        let _ = tokenizer.decode(&[token_id], true).map_err(E::msg)?;
        Ok(true)
    })?;
    let total = start.elapsed();
    let ttft = ttft.unwrap_or(total);
    let completion_tokens = generated.len();
    let tokens_per_second = if total.as_secs_f64() > 0.0 {
        completion_tokens as f64 / total.as_secs_f64()
    } else {
        0.0
    };

    Ok(RunResult {
        prompt_tokens: input_ids.len(),
        completion_tokens,
        ttft_ms: duration_ms(ttft),
        total_ms: duration_ms(total),
        tokens_per_second,
    })
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    benchmark: String,
    model: String,
    warmup_runs: usize,
    measured_runs: usize,
    prompt_tokens: usize,
    max_new_tokens: usize,
    temperature: f32,
    top_p: f32,
    repeat_penalty: f32,
    avg_ttft_ms: f64,
    avg_total_ms: f64,
    avg_completion_tokens: f64,
    avg_tokens_per_second: f64,
    runs: Vec<RunResult>,
}

impl BenchmarkReport {
    fn from_runs(
        benchmark: &str,
        model: &str,
        prompt_tokens: usize,
        _defaults: &SamplingDefaults,
        cfg: &GenerationConfig,
        warmup_runs: usize,
        runs: &[RunResult],
    ) -> Self {
        let measured_runs = runs.len();
        let avg_ttft_ms = average(runs.iter().map(|r| r.ttft_ms));
        let avg_total_ms = average(runs.iter().map(|r| r.total_ms));
        let avg_completion_tokens = average(runs.iter().map(|r| r.completion_tokens as f64));
        let avg_tokens_per_second = average(runs.iter().map(|r| r.tokens_per_second));

        Self {
            benchmark: benchmark.to_string(),
            model: model.to_string(),
            warmup_runs,
            measured_runs,
            prompt_tokens,
            max_new_tokens: cfg.max_new_tokens,
            temperature: cfg.temperature,
            top_p: cfg.top_p,
            repeat_penalty: cfg.repeat_penalty,
            avg_ttft_ms,
            avg_total_ms,
            avg_completion_tokens,
            avg_tokens_per_second,
            runs: runs.to_vec(),
        }
    }
}

#[derive(Debug, Serialize)]
struct SuiteReport {
    model: String,
    warmup_runs: usize,
    measured_runs: usize,
    reports: Vec<BenchmarkReport>,
}

fn run_benchmark(
    engine: &mut dyn fermi_runtime::InferenceEngine,
    tokenizer: &tokenizers::Tokenizer,
    device: &Device,
    max_ctx: usize,
    model_id: &str,
    sampling_defaults: &SamplingDefaults,
    gen_cfg: &GenerationConfig,
    cli: &CliConfig,
    benchmark: &str,
    prompt: String,
) -> Result<BenchmarkReport> {
    let prompt_tokens = tokenizer
        .encode(prompt.clone(), false)
        .map_err(E::msg)?
        .get_ids()
        .len();
    if prompt_tokens + gen_cfg.max_new_tokens + 8 > max_ctx {
        return Err(E::msg(format!(
            "prompt too long for benchmark '{}': {} tokens (limit {})",
            benchmark, prompt_tokens, max_ctx
        )));
    }

    for _ in 0..cli.warmup {
        let _ = run_once(engine, tokenizer, device, &prompt, gen_cfg)?;
    }

    let mut runs = Vec::with_capacity(cli.runs);
    for idx in 0..cli.runs {
        let result = run_once(engine, tokenizer, device, &prompt, gen_cfg)?;
        if !cli.json {
            println!(
                "[{}] run {:>2}: ttft={:.2} ms total={:.2} ms completion_tokens={} tok/s={:.2}",
                benchmark,
                idx + 1,
                result.ttft_ms,
                result.total_ms,
                result.completion_tokens,
                result.tokens_per_second
            );
        }
        runs.push(result);
    }

    Ok(BenchmarkReport::from_runs(
        benchmark,
        model_id,
        prompt_tokens,
        sampling_defaults,
        gen_cfg,
        cli.warmup,
        &runs,
    ))
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

fn print_report(report: &BenchmarkReport) {
    println!();
    println!("benchmark: {}", report.benchmark);
    println!("model: {}", report.model);
    println!("prompt_tokens: {}", report.prompt_tokens);
    println!("warmup_runs: {}", report.warmup_runs);
    println!("measured_runs: {}", report.measured_runs);
    println!(
        "sampling: max_new_tokens={} temperature={} top_p={} repeat_penalty={}",
        report.max_new_tokens, report.temperature, report.top_p, report.repeat_penalty
    );
    println!("avg_ttft_ms: {:.2}", report.avg_ttft_ms);
    println!("avg_total_ms: {:.2}", report.avg_total_ms);
    println!("avg_completion_tokens: {:.2}", report.avg_completion_tokens);
    println!("avg_tokens_per_second: {:.2}", report.avg_tokens_per_second);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_prompt_wraps_user_content() {
        let prompt = render_chat_prompt(ModelArch::Qwen, "hello", Some("sys"));
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn average_is_zero_for_empty_input() {
        assert_eq!(average(std::iter::empty()), 0.0);
    }

    #[test]
    fn suite_all_contains_expected_cases() {
        let cases = build_suite_cases("all", ModelArch::Qwen).expect("suite should build");
        let names: Vec<_> = cases.into_iter().map(|case| case.name).collect();
        assert_eq!(names, vec!["short", "medium", "long", "reasoning"]);
    }
}
