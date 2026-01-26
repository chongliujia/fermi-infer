use anyhow::{Error as E, Result};
use fermi_models::qwen3::Config;
use hf_hub::{api::sync::ApiBuilder, Cache, Repo, RepoType};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

pub struct Qwen3Files {
    pub tokenizer: PathBuf,
    pub config: PathBuf,
    pub weights: Vec<PathBuf>,
}

pub fn download_qwen3_files(model_repo_id: &str, allow_network: bool) -> Result<Qwen3Files> {
    if let Ok(dir) = std::env::var("FERMI_MODEL_DIR") {
        if let Some(files) = try_local_qwen3_files(Path::new(&dir))? {
            return Ok(files);
        }
    }
    if let Some(files) = try_local_qwen3_files(Path::new(model_repo_id))? {
        return Ok(files);
    }
    if let Some(files) = try_hf_cache_qwen3_files(model_repo_id)? {
        return Ok(files);
    }

    if !allow_network {
        return Err(E::msg("offline mode: local model files not found"));
    }

    let token = std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGINGFACE_HUB_TOKEN").ok());
    let api = ApiBuilder::from_env().with_token(token).build()?;
    let repo = api.repo(Repo::new(model_repo_id.to_string(), RepoType::Model));

    let tokenizer = repo.get("tokenizer.json")?;
    let config = repo.get("config.json")?;

    let weights = vec![
        repo.get("model-00001-of-00002.safetensors")?,
        repo.get("model-00002-of-00002.safetensors")?,
    ];

    Ok(Qwen3Files {
        tokenizer,
        config,
        weights,
    })
}

fn try_hf_cache_qwen3_files(model_repo_id: &str) -> Result<Option<Qwen3Files>> {
    let cache = Cache::from_env();
    let repo = cache.repo(Repo::new(model_repo_id.to_string(), RepoType::Model));

    let tokenizer = match repo.get("tokenizer.json") {
        Some(path) => path,
        None => return Ok(None),
    };
    let config = match repo.get("config.json") {
        Some(path) => path,
        None => return Ok(None),
    };

    let weights = if let Some(single) = repo.get("model.safetensors") {
        vec![single]
    } else {
        let shard1 = repo.get("model-00001-of-00002.safetensors");
        let shard2 = repo.get("model-00002-of-00002.safetensors");
        match (shard1, shard2) {
            (Some(s1), Some(s2)) => vec![s1, s2],
            _ => return Ok(None),
        }
    };

    Ok(Some(Qwen3Files {
        tokenizer,
        config,
        weights,
    }))
}

fn try_local_qwen3_files(dir: &Path) -> Result<Option<Qwen3Files>> {
    if !dir.is_dir() {
        return Ok(None);
    }

    let tokenizer = dir.join("tokenizer.json");
    let config = dir.join("config.json");
    if !tokenizer.is_file() || !config.is_file() {
        return Ok(None);
    }

    let single = dir.join("model.safetensors");
    let shard1 = dir.join("model-00001-of-00002.safetensors");
    let shard2 = dir.join("model-00002-of-00002.safetensors");

    let weights = if single.is_file() {
        vec![single]
    } else if shard1.is_file() && shard2.is_file() {
        vec![shard1, shard2]
    } else {
        return Err(E::msg("model weights not found in local directory"));
    };

    Ok(Some(Qwen3Files {
        tokenizer,
        config,
        weights,
    }))
}

pub fn load_tokenizer(path: impl AsRef<Path>) -> Result<Tokenizer> {
    Tokenizer::from_file(path).map_err(E::msg)
}

pub fn load_qwen3_config(path: impl AsRef<Path>) -> Result<Config> {
    let config_content = std::fs::read_to_string(path)?;
    let mut config_value: serde_json::Value = serde_json::from_str(&config_content)?;

    if let Some(obj) = config_value.as_object_mut() {
        let default_window = obj
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(32768);

        if let Some(sw) = obj.get("sliding_window") {
            if sw.is_null() {
                obj.insert(
                    "sliding_window".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(default_window)),
                );
            }
        }
    }

    let config: Config = serde_json::from_value(config_value)?;
    Ok(config)
}
