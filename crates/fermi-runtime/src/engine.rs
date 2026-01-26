use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use fermi_models::qwen3::{Config, Qwen3Model};
use crate::session::{SessionId, SessionStore};

pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub repeat_penalty: f32,
    pub stop_tokens: Vec<u32>,
    pub temperature: f32,
    pub top_p: f32,
}

pub struct PrefillOutput {
    pub next_token_id: u32,
    pub generated_ids: Vec<u32>,
    pub current_pos: usize,
}

// =========================================================================
// Original Qwen3Engine (Keeping it for backward compatibility for now)
// =========================================================================
pub struct Qwen3Engine {
    model: Qwen3Model,
}

impl Qwen3Engine {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let model = Qwen3Model::new(config, vb)?;
        Ok(Self { model })
    }

    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    pub fn prefill(
        &mut self,
        input_ids: &[u32],
        device: &Device,
        cfg: &GenerationConfig,
    ) -> Result<PrefillOutput> {
        self.prefill_with_offset(input_ids, 0, device, cfg)
    }

    pub fn prefill_with_offset(
        &mut self,
        input_ids: &[u32],
        offset: usize,
        device: &Device,
        cfg: &GenerationConfig,
    ) -> Result<PrefillOutput> {
        let input_tensor = Tensor::new(input_ids, device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input_tensor, offset)?;
        let (_b, seq_len, _vocab) = logits.dims3()?;
        let last_token_logits = logits.i((0, seq_len - 1, ..))?;

        let mut generated_ids = Vec::new();

        let mut rng = rand::thread_rng();
        let next_token_id = sample_token(&last_token_logits, cfg, &generated_ids, &mut rng)?;
        generated_ids.push(next_token_id);

        Ok(PrefillOutput {
            next_token_id,
            generated_ids,
            current_pos: offset + input_ids.len(),
        })
    }

    pub fn decode_step(
        &mut self,
        token_id: u32,
        current_pos: usize,
        generated_ids: &[u32],
        device: &Device,
        cfg: &GenerationConfig,
    ) -> Result<u32> {
        let input_tensor = Tensor::new(&[token_id], device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input_tensor, current_pos)?;
        let last_token_logits = logits.i((0, 0, ..))?;
        let mut rng = rand::thread_rng();
        sample_token(&last_token_logits, cfg, generated_ids, &mut rng)
    }

    pub fn generate_stream<F>(
        &mut self,
        input_ids: &[u32],
        device: &Device,
        cfg: &GenerationConfig,
        mut on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> Result<bool>,
    {
        if cfg.max_new_tokens == 0 {
            return Ok(Vec::new());
        }

        let prefill = self.prefill_with_offset(input_ids, 0, device, cfg)?;
        let mut next_token_id = prefill.next_token_id;
        let mut generated_ids = prefill.generated_ids;
        let mut current_pos = prefill.current_pos;
        if !on_token(next_token_id)? {
            return Ok(generated_ids);
        }

        if cfg.stop_tokens.contains(&next_token_id) {
            return Ok(generated_ids);
        }

        for _ in 0..cfg.max_new_tokens {
            next_token_id =
                self.decode_step(next_token_id, current_pos, &generated_ids, device, cfg)?;
            generated_ids.push(next_token_id);
            if !on_token(next_token_id)? {
                break;
            }

            if cfg.stop_tokens.contains(&next_token_id) {
                break;
            }

            current_pos += 1;
        }

        Ok(generated_ids)
    }

    pub fn generate_stream_with_offset<F>(
        &mut self,
        input_ids: &[u32],
        offset: usize,
        device: &Device,
        cfg: &GenerationConfig,
        mut on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> Result<bool>,
    {
        if cfg.max_new_tokens == 0 {
            return Ok(Vec::new());
        }

        let prefill = self.prefill_with_offset(input_ids, offset, device, cfg)?;
        let mut next_token_id = prefill.next_token_id;
        let mut generated_ids = prefill.generated_ids;
        let mut current_pos = prefill.current_pos;
        if !on_token(next_token_id)? {
            return Ok(generated_ids);
        }

        if cfg.stop_tokens.contains(&next_token_id) {
            return Ok(generated_ids);
        }

        for _ in 0..cfg.max_new_tokens {
            next_token_id =
                self.decode_step(next_token_id, current_pos, &generated_ids, device, cfg)?;
            generated_ids.push(next_token_id);
            if !on_token(next_token_id)? {
                break;
            }

            if cfg.stop_tokens.contains(&next_token_id) {
                break;
            }

            current_pos += 1;
        }

        Ok(generated_ids)
    }

    pub fn append_tokens(
        &mut self,
        tokens: &[u32],
        offset: usize,
        device: &Device,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let input_tensor = Tensor::new(tokens, device)?.unsqueeze(0)?;
        let _ = self.model.forward(&input_tensor, offset)?;
        Ok(())
    }

    pub fn generate_stream_with_session<F, S: SessionStore>(
        &mut self,
        session_store: &S,
        session_id: SessionId,
        input_ids: &[u32],
        device: &Device,
        cfg: &GenerationConfig,
        on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> Result<bool>,
    {
        let _state = session_store.get_or_create(session_id.clone());
        let out = self.generate_stream(input_ids, device, cfg, on_token)?;
        session_store.touch(&session_id);
        Ok(out)
    }
}

fn sample_token<R: rand::Rng + ?Sized>(
    logits: &Tensor,
    cfg: &GenerationConfig,
    context: &[u32],
    rng: &mut R,
) -> Result<u32> {
    if cfg.temperature <= 0.0 || cfg.top_p <= 0.0 {
        return Ok(logits.argmax(0)?.to_scalar::<u32>()?);
    }

    let logits_f32 = if logits.dtype() == DType::F32 {
        logits.clone()
    } else {
        logits.to_dtype(DType::F32)?
    };
    let mut logit_vec = logits_f32.to_vec1::<f32>()?;
    if cfg.repeat_penalty > 1.0 && !context.is_empty() {
        let start_index = if context.len() > 64 {
            context.len() - 64
        } else {
            0
        };
        for &token_id in &context[start_index..] {
            let idx = token_id as usize;
            if idx < logit_vec.len() {
                let v = logit_vec[idx];
                if v > 0.0 {
                    logit_vec[idx] = v / cfg.repeat_penalty;
                } else {
                    logit_vec[idx] = v * cfg.repeat_penalty;
                }
            }
        }
    }
    if (cfg.temperature - 1.0).abs() > f32::EPSILON {
        for v in &mut logit_vec {
            *v /= cfg.temperature;
        }
    }

    let mut max_logit = f32::NEG_INFINITY;
    for &v in &logit_vec {
        if v > max_logit {
            max_logit = v;
        }
    }
    let mut exp_vec = Vec::with_capacity(logit_vec.len());
    let mut sum = 0.0f32;
    for &v in &logit_vec {
        let ev = (v - max_logit).exp();
        exp_vec.push(ev);
        sum += ev;
    }
    if sum == 0.0 {
        return Ok(logits.argmax(0)?.to_scalar::<u32>()?);
    }

    if cfg.top_p >= 1.0 {
        let mut r = (rand::RngCore::next_u32(rng) as f32 / u32::MAX as f32) * sum;
        for (idx, ev) in exp_vec.iter().enumerate() {
            if r <= *ev {
                return Ok(idx as u32);
            }
            r -= *ev;
        }
        return Ok((exp_vec.len().saturating_sub(1)) as u32);
    }

    let mut probs: Vec<(usize, f32)> = exp_vec
        .iter()
        .enumerate()
        .map(|(i, &e)| (i, e / sum))
        .collect();

    let mut cutoff = probs.len();
    let mut cumulative = 0.0f32;
    const TOP_P_MAX_K: usize = 2048;
    if probs.len() > TOP_P_MAX_K {
        let k = TOP_P_MAX_K.min(probs.len());
        let nth = k.saturating_sub(1);
        probs.select_nth_unstable_by(nth, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let topk = &mut probs[..k];
        topk.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (idx, &(_, p)) in topk.iter().enumerate() {
            cumulative += p;
            if cumulative >= cfg.top_p {
                cutoff = idx + 1;
                break;
            }
        }
        if cumulative < cfg.top_p {
            probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            cutoff = probs.len();
            cumulative = 0.0f32;
            for (idx, &(_, p)) in probs.iter().enumerate() {
                cumulative += p;
                if cumulative >= cfg.top_p {
                    cutoff = idx + 1;
                    break;
                }
            }
            probs.truncate(cutoff);
        } else {
            probs.truncate(cutoff);
        }
    } else {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (idx, &(_, p)) in probs.iter().enumerate() {
            cumulative += p;
            if cumulative >= cfg.top_p {
                cutoff = idx + 1;
                break;
            }
        }
        probs.truncate(cutoff);
    }

    let total: f32 = probs.iter().map(|(_, p)| p).sum();
    let mut r = (rand::RngCore::next_u32(rng) as f32 / u32::MAX as f32) * total;
    for &(token_id, p) in &probs {
        if r <= p {
            return Ok(token_id as u32);
        }
        r -= p;
    }
    Ok(probs.last().map(|(i, _)| *i as u32).unwrap_or(0))
}
