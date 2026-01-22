use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{VarBuilder, linear_no_bias, Linear}; // ✅ 移除了未使用的 Activation
use serde::Deserialize;

// ================= Config 定义 =================
#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

// ================= Rotary Embedding (RoPE) =================
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
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?.to_dtype(dtype)?.reshape((max_seq_len, 1))?;
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

// ================= Attention 层 (无 Bias) =================
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: candle_nn::RmsNorm,
    k_norm: candle_nn::RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: RotaryEmbedding,
    cache_k: Option<Tensor>,
    cache_v: Option<Tensor>,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        
        let q_proj = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;
        let q_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        
        let rotary_emb = RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            cache_k: None,
            cache_v: None,
        })
    }

    fn forward(&mut self, x: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b, seq_len, _hidden) = x.dims3()?;

        if seqlen_offset == 0 {
            self.cache_k = None;
            self.cache_v = None;
        }
        
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offset)?;

        let cache_len = match &self.cache_k {
            Some(k_cache) => k_cache.dims4()?.2,
            None => 0,
        };

        let (k_cat, v_cat) = match (&self.cache_k, &self.cache_v) {
            (Some(k_cache), Some(v_cache)) if seqlen_offset > 0 => {
                let k = Tensor::cat(&[k_cache, &k], 2)?;
                let v = Tensor::cat(&[v_cache, &v], 2)?;
                (k, v)
            }
            _ => (k, v),
        };

        let y = if x.device().is_metal() && seq_len == 1 {
            candle_nn::ops::sdpa(&q, &k_cat, &v_cat, 1. / (self.head_dim as f32).sqrt(), 1.)?
        } else {
            let k_rep = self.repeat_kv(&k_cat)?;
            let v_rep = self.repeat_kv(&v_cat)?;

            let att = (q.matmul(&k_rep.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = if seq_len > 1 {
                let mask = self.causal_mask(seq_len, cache_len, att.dtype(), x.device())?;
                att.broadcast_add(&mask)?
            } else {
                att
            };
            let att = candle_nn::ops::softmax(&att, 3)?;
            att.matmul(&v_rep.contiguous()?)?
        };

        self.cache_k = Some(k_cat);
        self.cache_v = Some(v_cat);

        let y = y.transpose(1, 2)?.reshape((b, seq_len, ()))?;
        self.o_proj.forward(&y)
    }

    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            Ok(x.clone())
        } else {
            let (b, n_kv_head, seq_len, head_dim) = x.dims4()?;
            let x = x.unsqueeze(2)?.expand((b, n_kv_head, n_rep, seq_len, head_dim))?;
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
        self.cache_k = None;
        self.cache_v = None;
    }
}

// ================= MLP 层 (无 Bias) =================
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        
        let gate_proj = linear_no_bias(hidden, intermediate, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden, intermediate, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate, hidden, vb.pp("down_proj"))?;

        Ok(Self { gate_proj, up_proj, down_proj })
    }

    // ✅ 修复报错: 移除 x.prev_op()，修正计算逻辑
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_gate = self.gate_proj.forward(x)?;
        let x_gate = candle_nn::ops::silu(&x_gate)?;
        // 关键修正：up_proj 应该接受原始输入 x，而不是 gate 的输出
        let x_up = self.up_proj.forward(x)?; 
        let x = (x_gate * x_up)?;
        self.down_proj.forward(&x)
    }
}

// ================= Transformer Block =================
struct Block {
    rms_1: candle_nn::RmsNorm,
    attn: Attention,
    rms_2: candle_nn::RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let rms_1 = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let attn = Attention::new(cfg, vb.pp("self_attn"))?;
        let rms_2 = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        Ok(Self { rms_1, attn, rms_2, mlp })
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

// ================= 主模型定义 (Qwen3) =================
pub struct Qwen3Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Block>,
    norm: candle_nn::RmsNorm,
    lm_head: Linear,
}

impl Qwen3Model {
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
