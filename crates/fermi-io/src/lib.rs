use anyhow::{Error as E, Result};
use fermi_models::qwen3::Config;
use hf_hub::{Cache, Repo, RepoType, api::sync::ApiBuilder};
use std::collections::BTreeSet;
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

    let weights = if let Ok(single) = repo.get("model.safetensors") {
        vec![single]
    } else if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        let names = parse_safetensor_index(&index_path)?;
        if names.is_empty() {
            return Err(E::msg(
                "no safetensor shards found in model.safetensors.index.json",
            ));
        }
        let mut files = Vec::with_capacity(names.len());
        for name in names {
            files.push(repo.get(&name)?);
        }
        files
    } else {
        return Err(E::msg(
            "model weights not found: expected model.safetensors or model.safetensors.index.json",
        ));
    };

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
    } else if let Some(index_path) = repo.get("model.safetensors.index.json") {
        let names = parse_safetensor_index(&index_path)?;
        if names.is_empty() {
            return Ok(None);
        }
        let mut files = Vec::with_capacity(names.len());
        for name in names {
            let Some(path) = repo.get(&name) else {
                return Ok(None);
            };
            files.push(path);
        }
        files
    } else {
        return Ok(None);
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

    let weights = if single.is_file() {
        vec![single]
    } else {
        let shards = collect_local_shards(dir)?;
        if shards.is_empty() {
            return Err(E::msg("model weights not found in local directory"));
        }
        shards
    };

    Ok(Some(Qwen3Files {
        tokenizer,
        config,
        weights,
    }))
}

fn parse_safetensor_index(index_path: &Path) -> Result<Vec<String>> {
    let index_content = std::fs::read_to_string(index_path)?;
    let index_value: serde_json::Value = serde_json::from_str(&index_content)?;
    let weight_map = index_value
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| E::msg("invalid safetensor index: missing weight_map object"))?;

    let mut names = BTreeSet::new();
    for value in weight_map.values() {
        if let Some(file_name) = value.as_str() {
            if file_name.ends_with(".safetensors") {
                names.insert(file_name.to_string());
            }
        }
    }
    Ok(names.into_iter().collect())
}

fn collect_local_shards(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut shards: Vec<(usize, usize, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some((idx, total)) = parse_shard_name(&name) else {
            continue;
        };
        shards.push((idx, total, entry.path()));
    }

    if shards.is_empty() {
        return Ok(Vec::new());
    }

    shards.sort_by_key(|(idx, _, _)| *idx);
    let total = shards[0].1;
    for (i, (idx, shard_total, _)) in shards.iter().enumerate() {
        if *shard_total != total {
            return Err(E::msg("inconsistent shard totals in local model directory"));
        }
        if *idx != i + 1 {
            return Err(E::msg(
                "non-contiguous shard numbering in local model directory",
            ));
        }
    }
    if total != shards.len() {
        return Err(E::msg("incomplete shard set in local model directory"));
    }

    Ok(shards.into_iter().map(|(_, _, path)| path).collect())
}

fn parse_shard_name(name: &str) -> Option<(usize, usize)> {
    let body = name.strip_prefix("model-")?.strip_suffix(".safetensors")?;
    let (idx, total) = body.split_once("-of-")?;
    let idx = idx.parse::<usize>().ok()?;
    let total = total.parse::<usize>().ok()?;
    Some((idx, total))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(prefix: &str) -> Self {
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let path = std::env::temp_dir().join(format!("{prefix}-{}-{ts}", std::process::id()));
            fs::create_dir_all(&path).expect("create temp dir");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn parse_index_collects_unique_sorted_names() {
        let dir = TempDir::new("fermi-io-index");
        let path = dir.path().join("model.safetensors.index.json");
        let content = r#"{
            "weight_map": {
                "a": "model-00002-of-00003.safetensors",
                "b": "model-00001-of-00003.safetensors",
                "c": "model-00003-of-00003.safetensors",
                "d": "model-00001-of-00003.safetensors"
            }
        }"#;
        fs::write(&path, content).expect("write index");

        let names = parse_safetensor_index(&path).expect("parse index");
        assert_eq!(
            names,
            vec![
                "model-00001-of-00003.safetensors".to_string(),
                "model-00002-of-00003.safetensors".to_string(),
                "model-00003-of-00003.safetensors".to_string(),
            ]
        );
    }

    #[test]
    fn collect_local_shards_requires_contiguous_set() {
        let dir = TempDir::new("fermi-io-shards-missing");
        fs::write(dir.path().join("model-00001-of-00003.safetensors"), b"").expect("write shard1");
        fs::write(dir.path().join("model-00003-of-00003.safetensors"), b"").expect("write shard3");

        let err = collect_local_shards(dir.path()).expect_err("should fail");
        assert!(
            err.to_string().contains("non-contiguous") || err.to_string().contains("incomplete"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn try_local_files_supports_sharded_models() {
        let dir = TempDir::new("fermi-io-local-sharded");
        fs::write(dir.path().join("tokenizer.json"), b"{}").expect("write tokenizer");
        fs::write(dir.path().join("config.json"), b"{}").expect("write config");
        fs::write(dir.path().join("model-00002-of-00002.safetensors"), b"").expect("write shard2");
        fs::write(dir.path().join("model-00001-of-00002.safetensors"), b"").expect("write shard1");

        let files = try_local_qwen3_files(dir.path())
            .expect("scan local files")
            .expect("should find files");
        assert_eq!(files.weights.len(), 2);
        assert!(files.weights[0].ends_with("model-00001-of-00002.safetensors"));
        assert!(files.weights[1].ends_with("model-00002-of-00002.safetensors"));
    }
}
