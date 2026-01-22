// 引入我们自定义的模块（用来解决 Qwen3 无 Bias 的问题）
mod model_qwen3;
use model_qwen3::{Config, Qwen3Model};

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor, IndexOp};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::Tokenizer;
use std::io::Write;

fn main() -> Result<()> {
    // 1. 基础环境设置
    let device = device_setup()?;
    println!("🚀 运行设备: {:?}", device);

    // ==========================================
    // 指定 Qwen3 官方模型 ID
    // ==========================================
    let model_repo_id = "Qwen/Qwen3-4B";
    
    println!("📥 正在连接 HuggingFace: {} ...", model_repo_id);
    let api = ApiBuilder::from_env().build()?;
    let repo = api.repo(Repo::new(model_repo_id.to_string(), RepoType::Model));

    // 2. 下载基础文件
    println!("📥 下载 Config 和 Tokenizer...");
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let config_filename = repo.get("config.json")?;

    // ==========================================
    // 3. 下载权重 (Qwen3-1.7B 是分片文件)
    // ==========================================
    println!("📥 检测到模型为分片格式，开始下载权重...");
    let filenames = vec![
        repo.get("model-00001-of-00003.safetensors")?,
        repo.get("model-00002-of-00003.safetensors")?,
        repo.get("model-00003-of-00003.safetensors")?,
    ];
    println!("✅ 权重下载完成");

    // ==========================================
    // 4. Config 清洗 (防止 null 报错)
    // ==========================================
    println!("⚙️ 正在解析配置文件...");
    
    let config_content = std::fs::read_to_string(config_filename)?;
    let mut config_value: serde_json::Value = serde_json::from_str(&config_content)?;

    if let Some(obj) = config_value.as_object_mut() {
        // 获取上下文长度作为默认值
        let default_window = obj.get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(32768);

        // 修复 sliding_window 为 null 的情况
        if let Some(sw) = obj.get("sliding_window") {
            if sw.is_null() {
                println!("⚠️  Config 修复: 将 'sliding_window' 设为 {}", default_window);
                obj.insert(
                    "sliding_window".to_string(), 
                    serde_json::Value::Number(serde_json::Number::from(default_window))
                );
            }
        }
    }

    // ⚠️ 关键点：使用我们自定义的 Config 结构体解析
    let config: Config = serde_json::from_value(config_value)?;

    // 5. 加载权重
    let dtype = if device.is_metal() { DType::F16 } else { DType::F32 };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };

    // 6. 初始化模型 (使用自定义的 Qwen3Model)
    println!("🏗️ 正在构建模型架构 (Custom Qwen3 No-Bias)...");
    let mut model = Qwen3Model::new(&config, vb)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    model.clear_kv_cache();

    // ==========================================
    // 7. 构造 Prompt
    // ==========================================
    let raw_prompt = "你好，请解释一下什么是“第一性原理”。";
    let prompt = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        raw_prompt
    );
    println!("🗣️ Prompt: {:?}", prompt);

    let tokens = tokenizer.encode(prompt, false).map_err(E::msg)?;
    let input_ids = tokens.get_ids().to_vec();
    
    println!("🤖 开始推理...");

    // --- 阶段一：预填充 (Prefill) ---
    let mut input_tensor = Tensor::new(input_ids.clone(), &device)?.unsqueeze(0)?;
    let mut logits = model.forward(&input_tensor, 0)?; 
    let (_b, seq_len, _vocab) = logits.dims3()?;
    let mut last_token_logits = logits.i((0, seq_len - 1, ..))?;
    
    let mut generated_ids = vec![]; 
    apply_repeat_penalty(&mut last_token_logits, 1.1, &generated_ids)?;
    
    let mut next_token_id = last_token_logits.argmax(0)?.to_scalar::<u32>()?;
    generated_ids.push(next_token_id);

    print!("{}", tokenizer.decode(&[next_token_id], true).map_err(E::msg)?);
    std::io::stdout().flush()?;

    // --- 阶段二：解码循环 (Decode Loop) ---
    let max_new_tokens = 5096; 
    let mut current_pos = input_ids.len();

    for _ in 0..max_new_tokens {
        input_tensor = Tensor::new(&[next_token_id], &device)?.unsqueeze(0)?;
        // 注意：offset 是当前生成的总长度
        logits = model.forward(&input_tensor, current_pos)?; 
        last_token_logits = logits.i((0, 0, ..))?;

        apply_repeat_penalty(&mut last_token_logits, 1.1, &generated_ids)?;

        next_token_id = last_token_logits.argmax(0)?.to_scalar::<u32>()?;
        generated_ids.push(next_token_id);
        
        let token_text = tokenizer.decode(&[next_token_id], true).map_err(E::msg)?;
        print!("{}", token_text);
        std::io::stdout().flush()?;

        // 停止符检测
        if let Some(eos) = tokenizer.token_to_id("<|endoftext|>") { if next_token_id == eos { break; } }
        if let Some(im_end) = tokenizer.token_to_id("<|im_end|>") { if next_token_id == im_end { break; } }
        
        current_pos += 1;
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

fn apply_repeat_penalty(logits: &mut Tensor, penalty: f32, context: &[u32]) -> Result<()> {
    let device = logits.device();
    let orig_dtype = logits.dtype();
    let logits_f32 = if orig_dtype == DType::F32 {
        logits.clone()
    } else {
        logits.to_dtype(DType::F32)?
    };
    let mut logits_vec = logits_f32.to_vec1::<f32>()?;
    let start_index = if context.len() > 64 { context.len() - 64 } else { 0 };
    for &token_id in &context[start_index..] {
        let idx = token_id as usize;
        if idx < logits_vec.len() {
            let v = logits_vec[idx];
            if v > 0.0 {
                logits_vec[idx] = v / penalty;
            } else {
                logits_vec[idx] = v * penalty;
            }
        }
    }
    let mut out = Tensor::from_vec(logits_vec, logits.shape(), device)?;
    if orig_dtype != DType::F32 {
        out = out.to_dtype(orig_dtype)?;
    }
    *logits = out;
    Ok(())
}
