use anyhow::{Result, bail};

pub const DEFAULT_MAX_NEW_TOKENS: usize = 256;
pub const DEFAULT_MAX_NEW_TOKENS_CAP: usize = 9056;
pub const DEFAULT_TEMPERATURE: f32 = 0.2;
pub const DEFAULT_TOP_P: f32 = 1.0;
pub const DEFAULT_REPEAT_PENALTY: f32 = 1.1;

#[derive(Debug, Clone)]
pub struct SamplingDefaults {
    pub default_max_new_tokens: usize,
    pub max_new_tokens_cap: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
}

impl Default for SamplingDefaults {
    fn default() -> Self {
        Self {
            default_max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            max_new_tokens_cap: DEFAULT_MAX_NEW_TOKENS_CAP,
            temperature: DEFAULT_TEMPERATURE,
            top_p: DEFAULT_TOP_P,
            repeat_penalty: DEFAULT_REPEAT_PENALTY,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SamplingDefaultsOverride {
    pub default_max_new_tokens: Option<usize>,
    pub max_new_tokens_cap: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
}

pub fn sampling_defaults_from_env() -> Result<SamplingDefaults> {
    sampling_defaults_from_sources(SamplingDefaultsOverride::default())
}

pub fn sampling_defaults_from_sources(
    overrides: SamplingDefaultsOverride,
) -> Result<SamplingDefaults> {
    let mut defaults = SamplingDefaults::default();
    if let Some(v) = overrides.default_max_new_tokens {
        defaults.default_max_new_tokens = v;
    }
    if let Some(v) = overrides.max_new_tokens_cap {
        defaults.max_new_tokens_cap = v;
    }
    if let Some(v) = overrides.temperature {
        defaults.temperature = v;
    }
    if let Some(v) = overrides.top_p {
        defaults.top_p = v;
    }
    if let Some(v) = overrides.repeat_penalty {
        defaults.repeat_penalty = v;
    }

    if let Some(v) = env_usize("FERMI_DEFAULT_MAX_NEW_TOKENS")? {
        defaults.default_max_new_tokens = v;
    }
    if let Some(v) = env_usize("FERMI_MAX_NEW_TOKENS_CAP")? {
        defaults.max_new_tokens_cap = v;
    }
    if let Some(v) = env_f32("FERMI_DEFAULT_TEMPERATURE")? {
        defaults.temperature = v;
    }
    if let Some(v) = env_f32("FERMI_DEFAULT_TOP_P")? {
        defaults.top_p = v;
    }
    if let Some(v) = env_f32("FERMI_DEFAULT_REPEAT_PENALTY")? {
        defaults.repeat_penalty = v;
    }

    validate_sampling_defaults(&defaults)?;

    Ok(defaults)
}

fn validate_sampling_defaults(defaults: &SamplingDefaults) -> Result<()> {
    validate_max_new_tokens(defaults.default_max_new_tokens)?;
    validate_max_new_tokens(defaults.max_new_tokens_cap)?;
    if defaults.default_max_new_tokens > defaults.max_new_tokens_cap {
        bail!(
            "FERMI_DEFAULT_MAX_NEW_TOKENS ({}) cannot exceed FERMI_MAX_NEW_TOKENS_CAP ({})",
            defaults.default_max_new_tokens,
            defaults.max_new_tokens_cap
        );
    }
    validate_temperature(defaults.temperature)?;
    validate_top_p(defaults.top_p)?;
    validate_repeat_penalty(defaults.repeat_penalty)?;
    Ok(())
}

pub fn resolve_sampling_params(
    requested_max_new_tokens: Option<usize>,
    requested_temperature: Option<f32>,
    requested_top_p: Option<f32>,
    requested_repeat_penalty: Option<f32>,
    defaults: &SamplingDefaults,
) -> Result<SamplingParams> {
    let max_new_tokens = requested_max_new_tokens.unwrap_or(defaults.default_max_new_tokens);
    let temperature = requested_temperature.unwrap_or(defaults.temperature);
    let top_p = requested_top_p.unwrap_or(defaults.top_p);
    let repeat_penalty = requested_repeat_penalty.unwrap_or(defaults.repeat_penalty);

    validate_max_new_tokens(max_new_tokens)?;
    validate_temperature(temperature)?;
    validate_top_p(top_p)?;
    validate_repeat_penalty(repeat_penalty)?;

    Ok(SamplingParams {
        max_new_tokens: max_new_tokens.min(defaults.max_new_tokens_cap),
        temperature,
        top_p,
        repeat_penalty,
    })
}

fn validate_max_new_tokens(v: usize) -> Result<()> {
    if v == 0 {
        bail!("max_new_tokens must be > 0");
    }
    Ok(())
}

fn validate_temperature(v: f32) -> Result<()> {
    if !(0.0..=2.0).contains(&v) {
        bail!("temperature must be in [0.0, 2.0], got {}", v);
    }
    Ok(())
}

fn validate_top_p(v: f32) -> Result<()> {
    if !(v > 0.0 && v <= 1.0) {
        bail!("top_p must be in (0.0, 1.0], got {}", v);
    }
    Ok(())
}

fn validate_repeat_penalty(v: f32) -> Result<()> {
    if !(1.0..=2.0).contains(&v) {
        bail!("repeat_penalty must be in [1.0, 2.0], got {}", v);
    }
    Ok(())
}

fn env_usize(key: &str) -> Result<Option<usize>> {
    match std::env::var(key) {
        Ok(v) => v
            .parse::<usize>()
            .map(Some)
            .map_err(|e| anyhow::anyhow!("invalid {}='{}': {}", key, v, e)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(anyhow::anyhow!("cannot read {}: {}", key, e)),
    }
}

fn env_f32(key: &str) -> Result<Option<f32>> {
    match std::env::var(key) {
        Ok(v) => v
            .parse::<f32>()
            .map(Some)
            .map_err(|e| anyhow::anyhow!("invalid {}='{}': {}", key, v, e)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(anyhow::anyhow!("cannot read {}: {}", key, e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_with_defaults_when_request_missing() {
        let defaults = SamplingDefaults::default();
        let p = resolve_sampling_params(None, None, None, None, &defaults).expect("resolve");
        assert_eq!(p.max_new_tokens, DEFAULT_MAX_NEW_TOKENS);
        assert_eq!(p.temperature, DEFAULT_TEMPERATURE);
        assert_eq!(p.top_p, DEFAULT_TOP_P);
    }

    #[test]
    fn caps_max_new_tokens() {
        let defaults = SamplingDefaults {
            default_max_new_tokens: 256,
            max_new_tokens_cap: 128,
            temperature: 0.2,
            top_p: 1.0,
            repeat_penalty: 1.1,
        };
        let p = resolve_sampling_params(Some(512), None, None, None, &defaults).expect("resolve");
        assert_eq!(p.max_new_tokens, 128);
    }

    #[test]
    fn validates_ranges() {
        let defaults = SamplingDefaults::default();
        assert!(resolve_sampling_params(Some(0), None, None, None, &defaults).is_err());
        assert!(resolve_sampling_params(None, Some(-0.1), None, None, &defaults).is_err());
        assert!(resolve_sampling_params(None, None, Some(1.2), None, &defaults).is_err());
        assert!(resolve_sampling_params(None, None, None, Some(0.9), &defaults).is_err());
    }

    #[test]
    fn applies_source_overrides() {
        let defaults = sampling_defaults_from_sources(SamplingDefaultsOverride {
            default_max_new_tokens: Some(123),
            max_new_tokens_cap: Some(456),
            temperature: Some(0.6),
            top_p: Some(0.8),
            repeat_penalty: Some(1.2),
        })
        .expect("load defaults");
        assert_eq!(defaults.default_max_new_tokens, 123);
        assert_eq!(defaults.max_new_tokens_cap, 456);
        assert_eq!(defaults.temperature, 0.6);
        assert_eq!(defaults.top_p, 0.8);
        assert_eq!(defaults.repeat_penalty, 1.2);
    }
}
