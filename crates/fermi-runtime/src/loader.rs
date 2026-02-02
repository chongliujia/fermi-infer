use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use fermi_io::{
    ModelArch, ModelFiles, detect_model_arch, download_model_files, load_phi3_config,
    load_qwen_config,
};
use fermi_models::{phi3::Config as Phi3Config, qwen3::Config as QwenConfig};
use std::path::PathBuf;

use crate::engine::{InferenceEngine, Phi3Engine, Qwen3Engine};

pub struct ModelBuilder {
    files: ModelFiles,
    arch: ModelArch,
    config: LoadedConfig,
}

enum LoadedConfig {
    Qwen(QwenConfig),
    Phi3(Phi3Config),
}

impl ModelBuilder {
    pub fn new(model_id: &str, allow_network: bool) -> Result<Self> {
        let files = download_model_files(model_id, allow_network)?;
        let arch = detect_model_arch(&files.config)?;
        let config = match arch {
            ModelArch::Qwen => LoadedConfig::Qwen(load_qwen_config(&files.config)?),
            ModelArch::Phi3 => LoadedConfig::Phi3(load_phi3_config(&files.config)?),
        };
        Ok(Self {
            files,
            arch,
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

        match &self.config {
            LoadedConfig::Qwen(config) => Ok(Box::new(Qwen3Engine::new(config, vb)?)),
            LoadedConfig::Phi3(config) => Ok(Box::new(Phi3Engine::new(config, vb)?)),
        }
    }

    pub fn tokenizer_path(&self) -> PathBuf {
        self.files.tokenizer.clone()
    }

    pub fn model_arch(&self) -> ModelArch {
        self.arch
    }

    pub fn max_position_embeddings(&self) -> usize {
        match &self.config {
            LoadedConfig::Qwen(config) => config.max_position_embeddings,
            LoadedConfig::Phi3(config) => config.max_position_embeddings,
        }
    }
}
