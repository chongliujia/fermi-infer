use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use fermi_io::{
    ModelArch, ModelFiles, detect_model_arch, download_model_files, load_llama_config,
    load_phi3_config, load_qwen_config,
};
use fermi_models::{
    llama::Config as LlamaConfig, phi3::Config as Phi3Config, qwen3::Config as QwenConfig,
};
use std::path::{Path, PathBuf};

use crate::engine::{InferenceEngine, Phi3Engine, Qwen3Engine};

pub struct ModelBuilder {
    files: ModelFiles,
    registry: &'static RuntimeModelRegistryEntry,
    config: LoadedConfig,
}

enum LoadedConfig {
    Qwen(QwenConfig),
    Llama(LlamaConfig),
    Phi3(Phi3Config),
}

struct RuntimeModelRegistryEntry {
    arch: ModelArch,
    load_config: fn(&Path) -> Result<LoadedConfig>,
    create_engine: fn(&LoadedConfig, VarBuilder) -> Result<Box<dyn InferenceEngine>>,
    max_position_embeddings: fn(&LoadedConfig) -> usize,
}

const RUNTIME_MODEL_REGISTRY: &[RuntimeModelRegistryEntry] = &[
    RuntimeModelRegistryEntry {
        arch: ModelArch::Qwen,
        load_config: load_qwen_loaded_config,
        create_engine: create_qwen_engine,
        max_position_embeddings: qwen_max_position_embeddings,
    },
    RuntimeModelRegistryEntry {
        arch: ModelArch::Llama,
        load_config: load_llama_loaded_config,
        create_engine: create_llama_engine,
        max_position_embeddings: llama_max_position_embeddings,
    },
    RuntimeModelRegistryEntry {
        arch: ModelArch::Phi3,
        load_config: load_phi3_loaded_config,
        create_engine: create_phi3_engine,
        max_position_embeddings: phi3_max_position_embeddings,
    },
];

fn registry_entry(arch: ModelArch) -> Result<&'static RuntimeModelRegistryEntry> {
    RUNTIME_MODEL_REGISTRY
        .iter()
        .find(|entry| entry.arch == arch)
        .ok_or_else(|| anyhow::anyhow!("no runtime factory registered for {:?}", arch))
}

fn load_qwen_loaded_config(path: &Path) -> Result<LoadedConfig> {
    Ok(LoadedConfig::Qwen(load_qwen_config(path)?))
}

fn load_phi3_loaded_config(path: &Path) -> Result<LoadedConfig> {
    Ok(LoadedConfig::Phi3(load_phi3_config(path)?))
}

fn load_llama_loaded_config(path: &Path) -> Result<LoadedConfig> {
    Ok(LoadedConfig::Llama(load_llama_config(path)?))
}

fn create_qwen_engine(
    config: &LoadedConfig,
    vb: VarBuilder,
) -> Result<Box<dyn InferenceEngine>> {
    let LoadedConfig::Qwen(config) = config else {
        anyhow::bail!("runtime registry/config mismatch for Qwen")
    };
    Ok(Box::new(Qwen3Engine::new(config, vb)?))
}

fn create_phi3_engine(
    config: &LoadedConfig,
    vb: VarBuilder,
) -> Result<Box<dyn InferenceEngine>> {
    let LoadedConfig::Phi3(config) = config else {
        anyhow::bail!("runtime registry/config mismatch for Phi3")
    };
    Ok(Box::new(Phi3Engine::new(config, vb)?))
}

fn create_llama_engine(
    config: &LoadedConfig,
    vb: VarBuilder,
) -> Result<Box<dyn InferenceEngine>> {
    let LoadedConfig::Llama(config) = config else {
        anyhow::bail!("runtime registry/config mismatch for Llama")
    };
    Ok(Box::new(Qwen3Engine::new(config, vb)?))
}

fn qwen_max_position_embeddings(config: &LoadedConfig) -> usize {
    let LoadedConfig::Qwen(config) = config else {
        panic!("runtime registry/config mismatch for Qwen");
    };
    config.max_position_embeddings
}

fn phi3_max_position_embeddings(config: &LoadedConfig) -> usize {
    let LoadedConfig::Phi3(config) = config else {
        panic!("runtime registry/config mismatch for Phi3");
    };
    config.max_position_embeddings
}

fn llama_max_position_embeddings(config: &LoadedConfig) -> usize {
    let LoadedConfig::Llama(config) = config else {
        panic!("runtime registry/config mismatch for Llama");
    };
    config.max_position_embeddings
}

impl ModelBuilder {
    pub fn new(model_id: &str, allow_network: bool) -> Result<Self> {
        let files = download_model_files(model_id, allow_network)?;
        let arch = detect_model_arch(&files.config)?;
        let registry = registry_entry(arch)?;
        let config = (registry.load_config)(&files.config)?;
        Ok(Self {
            files,
            registry,
            config,
        })
    }

    pub fn create_engine(&self, device: &Device) -> Result<Box<dyn InferenceEngine>> {
        let dtype = if device.is_metal() {
            DType::F16
        } else {
            DType::F32
        };

        // Load weights (mmap)
        // Note: VarBuilder::from_mmaped_safetensors requires 'static lifetime for the path usually,
        // or we just pass the paths. internal implementation handles it.
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&self.files.weights, dtype, device)? };

        (self.registry.create_engine)(&self.config, vb)
    }

    pub fn tokenizer_path(&self) -> PathBuf {
        self.files.tokenizer.clone()
    }

    pub fn model_arch(&self) -> ModelArch {
        self.registry.arch
    }

    pub fn max_position_embeddings(&self) -> usize {
        (self.registry.max_position_embeddings)(&self.config)
    }
}
