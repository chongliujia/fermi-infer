use anyhow::{Context, Result, bail};
use std::env;
use std::path::{Path, PathBuf};

use crate::sampling::SamplingDefaultsOverride;

pub const DEFAULT_CONFIG_FILE: &str = "fermi.toml";

#[derive(Debug, Clone, Default)]
pub struct AppConfig {
    pub model: ModelConfig,
    pub generation: GenerationDefaultsConfig,
    pub cli: CliConfig,
    pub openai: OpenAiConfig,
    pub grpc: GrpcConfig,
}

#[derive(Debug, Clone, Default)]
pub struct ModelConfig {
    pub id: Option<String>,
    pub offline: Option<bool>,
}

#[derive(Debug, Clone, Default)]
pub struct GenerationDefaultsConfig {
    pub default_max_new_tokens: Option<usize>,
    pub max_new_tokens_cap: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
}

impl GenerationDefaultsConfig {
    pub fn to_sampling_overrides(&self) -> SamplingDefaultsOverride {
        SamplingDefaultsOverride {
            default_max_new_tokens: self.default_max_new_tokens,
            max_new_tokens_cap: self.max_new_tokens_cap,
            temperature: self.temperature,
            top_p: self.top_p,
            repeat_penalty: self.repeat_penalty,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct OpenAiConfig {
    pub addr: Option<String>,
    pub engine_pool: Option<usize>,
    pub default_system_prompt: Option<String>,
    pub default_system_prompt_file: Option<String>,
    pub default_thinking: Option<String>,
    pub supports_thinking: Option<bool>,
    pub disable_think: Option<bool>,
}

#[derive(Debug, Clone, Default)]
pub struct GrpcConfig {
    pub addr: Option<String>,
    pub engine_pool: Option<usize>,
    pub timeout_ms: Option<u64>,
    pub session_ttl_ms: Option<u64>,
    pub session_max: Option<usize>,
    pub default_system_prompt: Option<String>,
    pub default_system_prompt_file: Option<String>,
    pub disable_think: Option<bool>,
}

#[derive(Debug, Clone, Default)]
pub struct CliConfig {
    pub timeout_ms: Option<u64>,
    pub default_system_prompt: Option<String>,
    pub default_system_prompt_file: Option<String>,
    pub disable_think: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct LoadedConfig {
    pub path: Option<PathBuf>,
    pub config: AppConfig,
}

#[derive(Debug, Clone, Copy, Default)]
enum Section {
    #[default]
    Root,
    Model,
    Generation,
    Cli,
    OpenAi,
    Grpc,
}

pub fn load_config(explicit_path: Option<&str>) -> Result<LoadedConfig> {
    let path = resolve_config_path(explicit_path);
    match path {
        Some(path) => {
            let content = std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read config file '{}'", path.display()))?;
            let config = parse_config(&content)
                .with_context(|| format!("invalid config file '{}'", path.display()))?;
            Ok(LoadedConfig {
                path: Some(path),
                config,
            })
        }
        None => Ok(LoadedConfig {
            path: None,
            config: AppConfig::default(),
        }),
    }
}

impl LoadedConfig {
    pub fn resolve_path(&self, path: &str) -> PathBuf {
        let candidate = PathBuf::from(path);
        if candidate.is_absolute() {
            return candidate;
        }
        if let Some(config_path) = &self.path
            && let Some(parent) = config_path.parent()
        {
            return parent.join(candidate);
        }
        candidate
    }

    pub fn read_text_file(&self, path: &str) -> Result<String> {
        let resolved = self.resolve_path(path);
        std::fs::read_to_string(&resolved)
            .with_context(|| format!("failed to read text file '{}'", resolved.display()))
    }
}

fn resolve_config_path(explicit_path: Option<&str>) -> Option<PathBuf> {
    if let Some(path) = explicit_path {
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }

    if let Ok(path) = env::var("FERMI_CONFIG") {
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }

    let default_path = Path::new(DEFAULT_CONFIG_FILE);
    if default_path.is_file() {
        return Some(default_path.to_path_buf());
    }
    None
}

fn parse_config(input: &str) -> Result<AppConfig> {
    let mut cfg = AppConfig::default();
    let mut section = Section::Root;

    for (idx, raw_line) in input.lines().enumerate() {
        let line_no = idx + 1;
        let no_comment = strip_inline_comment(raw_line);
        let line = no_comment.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            let name = line[1..line.len() - 1].trim().to_ascii_lowercase();
            section = match name.as_str() {
                "model" => Section::Model,
                "generation" => Section::Generation,
                "cli" => Section::Cli,
                "openai" => Section::OpenAi,
                "grpc" => Section::Grpc,
                other => bail!("line {}: unknown section [{}]", line_no, other),
            };
            continue;
        }

        let Some((key, value)) = line.split_once('=') else {
            bail!("line {}: expected key = value", line_no);
        };
        let key = key.trim().to_ascii_lowercase();
        let value = value.trim();

        match section {
            Section::Root => match key.as_str() {
                "model" => cfg.model.id = Some(parse_string(value, line_no)?),
                "offline" => cfg.model.offline = Some(parse_bool(value, line_no)?),
                _ => bail!("line {}: unknown root key '{}'", line_no, key),
            },
            Section::Model => match key.as_str() {
                "id" => cfg.model.id = Some(parse_string(value, line_no)?),
                "offline" => cfg.model.offline = Some(parse_bool(value, line_no)?),
                _ => bail!("line {}: unknown model key '{}'", line_no, key),
            },
            Section::Generation => match key.as_str() {
                "default_max_new_tokens" => {
                    cfg.generation.default_max_new_tokens = Some(parse_usize(value, line_no)?)
                }
                "max_new_tokens_cap" => {
                    cfg.generation.max_new_tokens_cap = Some(parse_usize(value, line_no)?)
                }
                "temperature" => cfg.generation.temperature = Some(parse_f32(value, line_no)?),
                "top_p" => cfg.generation.top_p = Some(parse_f32(value, line_no)?),
                "repeat_penalty" => {
                    cfg.generation.repeat_penalty = Some(parse_f32(value, line_no)?)
                }
                _ => bail!("line {}: unknown generation key '{}'", line_no, key),
            },
            Section::Cli => match key.as_str() {
                "timeout_ms" => cfg.cli.timeout_ms = Some(parse_u64(value, line_no)?),
                "default_system_prompt" => {
                    cfg.cli.default_system_prompt = Some(parse_string(value, line_no)?)
                }
                "default_system_prompt_file" => {
                    cfg.cli.default_system_prompt_file = Some(parse_string(value, line_no)?)
                }
                "disable_think" => cfg.cli.disable_think = Some(parse_bool(value, line_no)?),
                _ => bail!("line {}: unknown cli key '{}'", line_no, key),
            },
            Section::OpenAi => match key.as_str() {
                "addr" => cfg.openai.addr = Some(parse_string(value, line_no)?),
                "engine_pool" => cfg.openai.engine_pool = Some(parse_usize(value, line_no)?),
                "default_system_prompt" => {
                    cfg.openai.default_system_prompt = Some(parse_string(value, line_no)?)
                }
                "default_system_prompt_file" => {
                    cfg.openai.default_system_prompt_file = Some(parse_string(value, line_no)?)
                }
                "default_thinking" => {
                    cfg.openai.default_thinking = Some(parse_string(value, line_no)?)
                }
                "supports_thinking" => {
                    cfg.openai.supports_thinking = Some(parse_bool(value, line_no)?)
                }
                "disable_think" => cfg.openai.disable_think = Some(parse_bool(value, line_no)?),
                _ => bail!("line {}: unknown openai key '{}'", line_no, key),
            },
            Section::Grpc => match key.as_str() {
                "addr" => cfg.grpc.addr = Some(parse_string(value, line_no)?),
                "engine_pool" => cfg.grpc.engine_pool = Some(parse_usize(value, line_no)?),
                "timeout_ms" => cfg.grpc.timeout_ms = Some(parse_u64(value, line_no)?),
                "session_ttl_ms" => cfg.grpc.session_ttl_ms = Some(parse_u64(value, line_no)?),
                "session_max" => cfg.grpc.session_max = Some(parse_usize(value, line_no)?),
                "default_system_prompt" => {
                    cfg.grpc.default_system_prompt = Some(parse_string(value, line_no)?)
                }
                "default_system_prompt_file" => {
                    cfg.grpc.default_system_prompt_file = Some(parse_string(value, line_no)?)
                }
                "disable_think" => cfg.grpc.disable_think = Some(parse_bool(value, line_no)?),
                _ => bail!("line {}: unknown grpc key '{}'", line_no, key),
            },
        }
    }

    Ok(cfg)
}

fn strip_inline_comment(line: &str) -> String {
    let mut out = String::new();
    let mut in_single = false;
    let mut in_double = false;

    for ch in line.chars() {
        match ch {
            '\'' if !in_double => {
                in_single = !in_single;
                out.push(ch);
            }
            '"' if !in_single => {
                in_double = !in_double;
                out.push(ch);
            }
            '#' if !in_single && !in_double => break,
            _ => out.push(ch),
        }
    }
    out
}

fn parse_string(raw: &str, line_no: usize) -> Result<String> {
    let s = raw.trim();
    if s.len() >= 2
        && ((s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')))
    {
        Ok(s[1..s.len() - 1].to_string())
    } else if s.is_empty() {
        bail!("line {}: empty string value", line_no)
    } else {
        Ok(s.to_string())
    }
}

fn parse_bool(raw: &str, line_no: usize) -> Result<bool> {
    let s = parse_string(raw, line_no)?.trim().to_ascii_lowercase();
    match s.as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => bail!("line {}: invalid bool '{}'", line_no, raw),
    }
}

fn parse_usize(raw: &str, line_no: usize) -> Result<usize> {
    parse_string(raw, line_no)?
        .parse::<usize>()
        .map_err(|e| anyhow::anyhow!("line {}: invalid usize '{}': {}", line_no, raw, e))
}

fn parse_u64(raw: &str, line_no: usize) -> Result<u64> {
    parse_string(raw, line_no)?
        .parse::<u64>()
        .map_err(|e| anyhow::anyhow!("line {}: invalid u64 '{}': {}", line_no, raw, e))
}

fn parse_f32(raw: &str, line_no: usize) -> Result<f32> {
    parse_string(raw, line_no)?
        .parse::<f32>()
        .map_err(|e| anyhow::anyhow!("line {}: invalid f32 '{}': {}", line_no, raw, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_prompt_file_keys() {
        let cfg = parse_config(
            r#"
[cli]
default_system_prompt_file = "prompts/cli.txt"

[openai]
default_system_prompt_file = "prompts/openai.txt"

[grpc]
default_system_prompt_file = "prompts/grpc.txt"
"#,
        )
        .expect("parse");
        assert_eq!(
            cfg.cli.default_system_prompt_file.as_deref(),
            Some("prompts/cli.txt")
        );
        assert_eq!(
            cfg.openai.default_system_prompt_file.as_deref(),
            Some("prompts/openai.txt")
        );
        assert_eq!(
            cfg.grpc.default_system_prompt_file.as_deref(),
            Some("prompts/grpc.txt")
        );
    }

    #[test]
    fn resolves_relative_path_from_config_dir() {
        let loaded = LoadedConfig {
            path: Some(PathBuf::from("/tmp/fermi/fermi.toml")),
            config: AppConfig::default(),
        };
        assert_eq!(
            loaded.resolve_path("prompts/system.txt"),
            PathBuf::from("/tmp/fermi/prompts/system.txt")
        );
    }
}
