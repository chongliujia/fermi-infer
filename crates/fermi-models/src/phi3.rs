use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder, linear_no_bias};
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / (cfg.rope_theta as f32).powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn forward(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

struct Attention {
    qkv_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: RotaryEmbedding,
    cache_k: Vec<Tensor>,
    cache_v: Vec<Tensor>,
    cache_k_tail: Option<Tensor>,
    cache_v_tail: Option<Tensor>,
    cache_len: usize,
}

impl Attention {
    const KV_CHUNK_SIZE: usize = 128;

    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let qkv_width = (num_heads + 2 * num_kv_heads) * head_dim;

        let qkv_proj = linear_no_bias(hidden_size, qkv_width, vb.pp("qkv_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;
        let rotary_emb = RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?;

        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            cache_k: Vec::new(),
            cache_v: Vec::new(),
            cache_k_tail: None,
            cache_v_tail: None,
            cache_len: 0,
        })
    }

    fn forward(&mut self, x: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b, seq_len, _hidden) = x.dims3()?;

        if seqlen_offset == 0 {
            self.clear_cache();
        }

        let qkv = self.qkv_proj.forward(x)?;
        let qkv = qkv.reshape((
            b,
            seq_len,
            self.num_heads + 2 * self.num_kv_heads,
            self.head_dim,
        ))?;

        let q = qkv.narrow(2, 0, self.num_heads)?.transpose(1, 2)?;
        let k = qkv
            .narrow(2, self.num_heads, self.num_kv_heads)?
            .transpose(1, 2)?;
        let v = qkv
            .narrow(2, self.num_heads + self.num_kv_heads, self.num_kv_heads)?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offset)?;

        let prev_cache_len = self.cache_len;
        self.append_kv(&k, &v)?;

        let y = if seq_len == 1 {
            let mut segments: Vec<(&Tensor, &Tensor)> = Vec::new();
            for (k_seg, v_seg) in self.cache_k.iter().zip(self.cache_v.iter()) {
                segments.push((k_seg, v_seg));
            }
            if let (Some(k_tail), Some(v_tail)) = (&self.cache_k_tail, &self.cache_v_tail) {
                segments.push((k_tail, v_tail));
            }

            if x.device().is_metal() && segments.len() == 1 {
                let (k_seg, v_seg) = segments[0];
                let k_rep = self.repeat_kv(k_seg)?;
                let v_rep = self.repeat_kv(v_seg)?;
                candle_nn::ops::sdpa(&q, &k_rep, &v_rep, 1. / (self.head_dim as f32).sqrt(), 1.)?
            } else {
                self.segmented_attention(&q, &segments)?
            }
        } else {
            let (k_cat, v_cat) = self.concat_cache()?;
            let k_rep = self.repeat_kv(&k_cat)?;
            let v_rep = self.repeat_kv(&v_cat)?;
            let q_mat = q.contiguous()?;
            let k_t = k_rep.transpose(2, 3)?.contiguous()?;
            let att = (q_mat.matmul(&k_t)? / (self.head_dim as f64).sqrt())?;
            let att = if seq_len > 1 {
                let mask = self.causal_mask(seq_len, prev_cache_len, att.dtype(), x.device())?;
                att.broadcast_add(&mask)?
            } else {
                att
            };
            let att = candle_nn::ops::softmax(&att, 3)?;
            att.matmul(&v_rep.contiguous()?)?
        };

        let y = y.transpose(1, 2)?.reshape((b, seq_len, ()))?;
        self.o_proj.forward(&y)
    }

    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            Ok(x.clone())
        } else {
            let (b, n_kv_head, seq_len, head_dim) = x.dims4()?;
            let x = x
                .unsqueeze(2)?
                .expand((b, n_kv_head, n_rep, seq_len, head_dim))?;
            x.reshape((b, n_kv_head * n_rep, seq_len, head_dim))
        }
    }

    fn causal_mask(
        &self,
        seq_len: usize,
        cache_len: usize,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let total_len = cache_len + seq_len;
        let mut mask = Vec::with_capacity(seq_len * total_len);
        for i in 0..seq_len {
            let limit = cache_len + i;
            for j in 0..total_len {
                mask.push(if j > limit { f32::NEG_INFINITY } else { 0. });
            }
        }
        let mask = Tensor::from_vec(mask, (seq_len, total_len), dev)?.to_dtype(dtype)?;
        Ok(mask.unsqueeze(0)?.unsqueeze(0)?)
    }

    fn clear_cache(&mut self) {
        self.cache_k.clear();
        self.cache_v.clear();
        self.cache_k_tail = None;
        self.cache_v_tail = None;
        self.cache_len = 0;
    }

    fn append_kv(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let seq_len = k.dims4()?.2;
        let mut start = 0usize;

        while start < seq_len {
            let remaining = seq_len - start;
            let cur_len = match &self.cache_k_tail {
                Some(t) => t.dims4()?.2,
                None => 0,
            };
            let space = Self::KV_CHUNK_SIZE.saturating_sub(cur_len);
            let take = remaining.min(space);

            let k_slice = k.narrow(2, start, take)?;
            let v_slice = v.narrow(2, start, take)?;

            let (new_k, new_v) = match (&self.cache_k_tail, &self.cache_v_tail) {
                (Some(k_tail), Some(v_tail)) => {
                    let k_next = Tensor::cat(&[k_tail, &k_slice], 2)?;
                    let v_next = Tensor::cat(&[v_tail, &v_slice], 2)?;
                    (k_next, v_next)
                }
                _ => (k_slice, v_slice),
            };

            self.cache_k_tail = Some(new_k);
            self.cache_v_tail = Some(new_v);
            self.cache_len += take;

            if cur_len + take == Self::KV_CHUNK_SIZE
                && let (Some(k_full), Some(v_full)) =
                    (self.cache_k_tail.take(), self.cache_v_tail.take())
            {
                self.cache_k.push(k_full);
                self.cache_v.push(v_full);
            }

            start += take;
        }

        Ok(())
    }

    fn concat_cache(&self) -> Result<(Tensor, Tensor)> {
        let mut k_list: Vec<&Tensor> = self.cache_k.iter().collect();
        let mut v_list: Vec<&Tensor> = self.cache_v.iter().collect();
        if let (Some(k_tail), Some(v_tail)) = (&self.cache_k_tail, &self.cache_v_tail) {
            k_list.push(k_tail);
            v_list.push(v_tail);
        }

        if k_list.is_empty() {
            candle_core::bail!("kv cache is empty");
        }

        let k_cat = if k_list.len() == 1 {
            k_list[0].clone()
        } else {
            Tensor::cat(&k_list, 2)?
        };
        let v_cat = if v_list.len() == 1 {
            v_list[0].clone()
        } else {
            Tensor::cat(&v_list, 2)?
        };

        Ok((k_cat, v_cat))
    }

    fn segmented_attention(&self, q: &Tensor, segments: &[(&Tensor, &Tensor)]) -> Result<Tensor> {
        let q_mat = q.contiguous()?;
        let mut scores: Vec<Tensor> = Vec::with_capacity(segments.len());
        let mut max_per_segment: Option<Tensor> = None;
        for (k_seg, _) in segments {
            let k_rep = self.repeat_kv(k_seg)?;
            let k_t = k_rep.transpose(2, 3)?.contiguous()?;
            let seg_scores = (q_mat.matmul(&k_t)? / (self.head_dim as f64).sqrt())?;
            let seg_max = seg_scores.max_keepdim(3)?;
            max_per_segment = Some(match max_per_segment {
                Some(m) => m.maximum(&seg_max)?,
                None => seg_max,
            });
            scores.push(seg_scores);
        }

        let max_per_segment = match max_per_segment {
            Some(m) => m,
            None => candle_core::bail!("no kv segments for attention"),
        };

        let mut exp_scores: Vec<Tensor> = Vec::with_capacity(scores.len());
        let mut denom: Option<Tensor> = None;
        for seg_scores in scores {
            let exp = seg_scores.broadcast_sub(&max_per_segment)?.exp()?;
            let seg_sum = exp.sum_keepdim(3)?;
            denom = Some(match denom {
                Some(d) => d.broadcast_add(&seg_sum)?,
                None => seg_sum,
            });
            exp_scores.push(exp);
        }

        let denom = match denom {
            Some(d) => d,
            None => candle_core::bail!("failed to compute attention normalization"),
        };

        let mut output: Option<Tensor> = None;
        for (exp, (_, v_seg)) in exp_scores.into_iter().zip(segments.iter()) {
            let v_rep = self.repeat_kv(v_seg)?.contiguous()?;
            let weight = exp.broadcast_div(&denom)?;
            let seg_out = weight.matmul(&v_rep)?;
            output = Some(match output {
                Some(o) => o.broadcast_add(&seg_out)?,
                None => seg_out,
            });
        }

        output.ok_or_else(|| candle_core::Error::Msg("empty attention output".into()))
    }
}

struct Mlp {
    gate_up_proj: Linear,
    down_proj: Linear,
    intermediate_size: usize,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;

        let gate_up_proj = linear_no_bias(hidden, 2 * intermediate, vb.pp("gate_up_proj"))?;
        let down_proj = linear_no_bias(intermediate, hidden, vb.pp("down_proj"))?;

        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size: intermediate,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let fused = self.gate_up_proj.forward(x)?;
        let gate = fused.narrow(2, 0, self.intermediate_size)?;
        let up = fused.narrow(2, self.intermediate_size, self.intermediate_size)?;
        let x = (candle_nn::ops::silu(&gate)? * up)?;
        self.down_proj.forward(&x)
    }
}

struct Block {
    rms_1: candle_nn::RmsNorm,
    attn: Attention,
    rms_2: candle_nn::RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let rms_1 =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let attn = Attention::new(cfg, vb.pp("self_attn"))?;
        let rms_2 = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        })
    }

    fn forward(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = self.attn.forward(&x, offset)?;
        let x = (x + residual)?;

        let residual = &x;
        let x = self.rms_2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = (x + residual)?;
        Ok(x)
    }

    fn clear_cache(&mut self) {
        self.attn.clear_cache();
    }
}

pub struct Phi3Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Block>,
    norm: candle_nn::RmsNorm,
    lm_head: Linear,
}

impl Phi3Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Block::new(cfg, vb_layers.pp(i))?);
        }
        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::new(embed_tokens.embeddings().clone(), None)
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        let mut x = self.embed_tokens.forward(input_ids)?;

        for layer in &mut self.layers {
            x = layer.forward(&x, seqlen_offset)?;
        }

        let x = self.norm.forward(&x)?;
        let x = x.narrow(1, seq_len - 1, 1)?;
        self.lm_head.forward(&x)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }
}
