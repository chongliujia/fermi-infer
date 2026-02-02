use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use fermi_io::{Qwen3Files, download_qwen3_files, load_qwen3_config};
use fermi_models::qwen3::Config;
use std::path::PathBuf;

use crate::engine::{InferenceEngine, Qwen3Engine};

pub struct ModelBuilder {
    files: Qwen3Files,
    config: Config,
}

impl ModelBuilder {
    pub fn new(model_id: &str, offline: bool) -> Result<Self> {
        // Future: detect architecture from config.json or model_id
        let files = download_qwen3_files(model_id, !offline)?;
        let config = load_qwen3_config(&files.config)?;
        Ok(Self { files, config })
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

        let engine = Qwen3Engine::new(&self.config, vb)?;
        Ok(Box::new(engine))
    }

    pub fn tokenizer_path(&self) -> PathBuf {
        self.files.tokenizer.clone()
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.config.max_position_embeddings
    }
}
