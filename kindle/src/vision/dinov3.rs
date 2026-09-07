// Adapted from kvark/dinovision at dc35cdf1c7c910cdd93c5b5362846842ae469a21 (MIT).
//! DINOv3 ViT encoder, expressed directly in meganeura's graph IR.
//!
//! Built by hand rather than imported: `meganeura::load_onnx` currently
//! recognizes only `Add`, `Gemm`, `MatMul`, and `Relu`, which is nowhere
//! near enough for a ViT. Every op this needs already exists in the IR,
//! so the graph is a faithful transcription of
//! `transformers/models/dinov3_vit/modeling_dinov3_vit.py`.
//!
//! Three things distinguish DINOv3 from a garden-variety ViT, and each is
//! handled below:
//!
//! 1. **No learned position embedding.** Position enters only as 2D axial
//!    RoPE applied to Q and K inside every layer.
//! 2. **Prefix tokens.** A CLS token and `num_register_tokens` register
//!    tokens are prepended to the patch tokens, and RoPE deliberately
//!    skips them.
//! 3. **LayerScale.** Each residual branch is scaled by a learned
//!    per-channel vector before being added back.

use super::{stack_streams, stream_rows};
use meganeura::{Graph, NodeId};

/// Architecture hyperparameters, mirroring HuggingFace's `DINOv3ViTConfig`.
#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: u32,
    pub intermediate_size: usize,
    pub num_register_tokens: usize,
    pub patch_size: usize,
    /// Side length in pixels of the (square) input the graph is compiled
    /// for. RoPE lets the real model take any size, but a meganeura graph
    /// has static shapes, so one session serves one resolution.
    pub image_size: usize,
    pub layer_norm_eps: f32,
    pub rope_theta: f32,
    /// `false` selects the plain GELU MLP; DINOv3 only gates on ViT-L and
    /// larger. Gated MLPs are not implemented here yet.
    pub use_gated_mlp: bool,
    /// Store the projection weights as f16, halving weight traffic from
    /// 86 MB to 43 MB per forward pass.
    ///
    /// Storage only — meganeura's generated matmul reads `array<f16>` and
    /// converts to f32 for the arithmetic, so this buys bandwidth, not ALU
    /// rate. Whether that helps depends on which of the two the device is
    /// short of; measure with `examples/bench` rather than assuming.
    ///
    /// Norms, biases, and LayerScale stay f32: they are a rounding error in
    /// size and the most precision-sensitive parts of the network.
    pub f16_weights: bool,
}

impl Config {
    /// `facebook/dinov3-vits16-pretrain-lvd1689m` — 21M parameters.
    pub fn vits16() -> Self {
        Self {
            hidden_size: 384,
            num_hidden_layers: 12,
            num_attention_heads: 6,
            intermediate_size: 1536,
            num_register_tokens: 4,
            patch_size: 16,
            image_size: 224,
            layer_norm_eps: 1e-5,
            rope_theta: 100.0,
            use_gated_mlp: false,
            f16_weights: false,
        }
    }

    /// Enable f16 weight storage. See [`Config::f16_weights`].
    pub fn with_f16_weights(mut self, enabled: bool) -> Self {
        self.f16_weights = enabled;
        self
    }

    /// Keep only the first `n` transformer layers.
    ///
    /// The deployment case study uses three layers to bound mobile work. Its
    /// decoder is trained specifically against that representation; decoder
    /// weights are not interchangeable across depths. `n = 0` is retained as
    /// an embedding-plus-final-norm diagnostic for reference validation.
    pub fn with_layers(mut self, n: usize) -> Self {
        assert!(n <= 12, "DINOv3 ViT-S/16 has 12 layers, not {n}");
        self.num_hidden_layers = n;
        self
    }

    /// Same weights, different input resolution. The patch grid scales
    /// with it, so cost grows quadratically — 224 gives a 14×14 grid at
    /// ~4.6 GMAC, 448 gives 28×28 at ~4× that.
    pub fn at_resolution(mut self, image_size: usize) -> Self {
        assert_eq!(
            image_size % self.patch_size,
            0,
            "image_size {image_size} is not a multiple of patch_size {}",
            self.patch_size
        );
        self.image_size = image_size;
        self
    }

    pub fn head_dim(&self) -> u32 {
        self.hidden_size as u32 / self.num_attention_heads
    }

    /// Patches along one side of the image — the feature grid resolution.
    pub fn grid(&self) -> usize {
        self.image_size / self.patch_size
    }

    pub fn num_patches(&self) -> usize {
        self.grid() * self.grid()
    }

    /// CLS + register tokens, which sit ahead of the patch tokens.
    pub fn num_prefix_tokens(&self) -> usize {
        1 + self.num_register_tokens
    }

    pub fn num_tokens(&self) -> usize {
        self.num_prefix_tokens() + self.num_patches()
    }

    /// Length of one flattened patch: `3 * patch_size²`.
    pub fn patch_dim(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }

    /// Multiply-accumulate count for one forward pass, for sanity-checking
    /// measured throughput against achievable device throughput.
    pub fn forward_macs(&self) -> u64 {
        let t = self.num_tokens() as u64;
        let d = self.hidden_size as u64;
        let ff = self.intermediate_size as u64;
        let per_layer =
            // Q, K, V, and output projections
            4 * t * d * d
            // scores and the value-weighted sum
            + 2 * t * t * d
            // MLP up and down
            + 2 * t * d * ff;
        let patch_projection = self.num_patches() as u64 * self.patch_dim() as u64 * d;
        self.num_hidden_layers as u64 * per_layer + patch_projection
    }
}

/// Precomputed 2D axial RoPE tables, each `[num_tokens, hidden_size]`.
///
/// Two deviations from a literal transcription, both deliberate:
///
/// * Prefix rows are `(cos, sin) = (1, 0)`, making the rotation an
///   identity there. DINOv3 skips RoPE on CLS and register tokens by
///   slicing the token dimension; encoding it as data instead keeps the
///   graph free of token-dimension splits.
/// * The per-head table is replicated across all heads, so the tables
///   multiply Q and K elementwise with no per-head indexing.
///
/// The coordinate jitter/shift/rescale in the reference is training-only
/// (`if self.training`), so `pos_embed_rescale` is correctly ignored here.
pub fn rope_tables(config: &Config) -> (Vec<f32>, Vec<f32>) {
    let head_dim = config.head_dim() as usize;
    // inv_freq has head_dim/4 entries: half the rotated pairs go to the
    // y axis and half to x.
    let quarter = head_dim / 4;
    let half = head_dim / 2;
    let heads = config.num_attention_heads as usize;
    let hidden = config.hidden_size;
    let grid = config.grid();

    // inv_freq = 1 / theta^linspace(0, 1, head_dim/4)
    let inv_freq: Vec<f32> = (0..quarter)
        .map(|i| 1.0 / config.rope_theta.powf(i as f32 * 4.0 / head_dim as f32))
        .collect();

    let mut cos = vec![0.0f32; config.num_tokens() * hidden];
    let mut sin = vec![0.0f32; config.num_tokens() * hidden];

    // Identity rotation over the prefix tokens.
    for t in 0..config.num_prefix_tokens() {
        for c in 0..hidden {
            cos[t * hidden + c] = 1.0;
            sin[t * hidden + c] = 0.0;
        }
    }

    let mut angles = vec![0.0f32; half];
    for py in 0..grid {
        for px in 0..grid {
            // Patch-center coordinates normalized to [-1, +1]. The
            // reference stacks them (y, x), and that order decides which
            // half of the rotated pairs encodes which axis.
            let y = 2.0 * ((py as f32 + 0.5) / grid as f32) - 1.0;
            let x = 2.0 * ((px as f32 + 0.5) / grid as f32) - 1.0;
            for i in 0..quarter {
                angles[i] = std::f32::consts::TAU * y * inv_freq[i];
                angles[quarter + i] = std::f32::consts::TAU * x * inv_freq[i];
            }

            let t = config.num_prefix_tokens() + py * grid + px;
            let row = t * hidden;
            for h in 0..heads {
                for d in 0..head_dim {
                    // `angles.tile(2)`: the upper half repeats the lower,
                    // which is what pairs dimension d with d + head_dim/2
                    // under `rotate_half`.
                    let a = angles[d % half];
                    cos[row + h * head_dim + d] = a.cos();
                    sin[row + h * head_dim + d] = a.sin();
                }
            }
        }
    }

    (cos, sin)
}

/// `x * cos + rotate_half(x) * sin`, where
/// `rotate_half([x1, x2]) = [-x2, x1]` splits each head's `head_dim` in
/// two.
///
/// Q and K arrive from `matmul` as `[tokens, heads * head_dim]`, and
/// meganeura's attention reads head `h` of token `t` at
/// `t * heads * head_dim + h * head_dim` — head-major, so each head's
/// slice is contiguous. That lets the split treat the tensor as
/// `tokens * heads` independent blocks of `head_dim`, with no transpose.
fn apply_rope(
    g: &mut Graph,
    x: NodeId,
    cos: NodeId,
    sin: NodeId,
    tokens: usize,
    heads: u32,
    head_dim: u32,
) -> NodeId {
    let hidden = (heads * head_dim) as usize;
    let blocks = tokens as u32 * heads;
    let half = head_dim / 2;

    let x1 = g.split_a(x, blocks, half, half, 1);
    let x2 = g.split_b(x, blocks, half, half, 1);
    let neg_x2 = g.neg(x2);
    let rotated = g.concat(neg_x2, x1, blocks, half, half, 1);
    let rotated = g.reshape(rotated, &[tokens, hidden]);

    let straight = g.mul(x, cos);
    let crossed = g.mul(rotated, sin);
    g.add(straight, crossed)
}

/// Apply a learned gain to the trailing hidden dimension of a
/// `[tokens, hidden]` matrix.
///
/// Meganeura's `mul_per_channel` follows NCHW layout and indexes its gate as
/// `linear_index / spatial`; using it directly with `spatial = 1` would read
/// beyond the hidden-sized gate after the first token. Transposing to
/// `[hidden, tokens]` makes each hidden coordinate an NCHW channel, with all
/// tokens as its spatial extent, and therefore gives the LayerScale broadcast
/// used by the reference implementation.
fn layer_scale(g: &mut Graph, x: NodeId, gain: NodeId, tokens: usize, hidden: usize) -> NodeId {
    let hidden_by_tokens = g.transpose(x);
    // `mul_per_channel` is an NCHW-flat operator. Keeping the two-dimensional
    // transpose shape would make its compiler interpret only the first axis
    // as the element count.
    let flat = g.reshape(hidden_by_tokens, &[hidden * tokens]);
    let scaled = g.mul_per_channel(flat, gain, hidden as u32, tokens as u32);
    let scaled = g.reshape(scaled, &[hidden, tokens]);
    g.transpose(scaled)
}

/// Build the encoder and return the final `[num_tokens, hidden_size]`
/// feature node.
///
/// Declares one input, `"patches"`, of shape `[num_patches, patch_dim]` —
/// see [`crate::vision::preprocess`] for the layout it expects. Parameter names
/// match the HuggingFace checkpoint so [`crate::vision::weights`] can bind them
/// directly.
///
/// Row 0 of the output is the CLS token, rows `1..=num_register_tokens`
/// are the registers, and the patch features follow in row-major grid
/// order.
pub fn build_encoder(g: &mut Graph, config: &Config) -> NodeId {
    build_batched_encoder(g, config, 1)
}

/// Stream-major images: dense layers share one weight set, but each image
/// attends only to its own prefix and patches. Returns `[streams * tokens, hidden]`.
pub fn build_batched_encoder(g: &mut Graph, config: &Config, streams: usize) -> NodeId {
    assert!(streams > 0, "DINOv3 needs at least one stream");
    assert!(
        !config.use_gated_mlp,
        "gated MLP (ViT-L and larger) is not implemented"
    );

    let hidden = config.hidden_size;
    let eps = config.layer_norm_eps;
    let heads = config.num_attention_heads;
    let head_dim = config.head_dim();
    let tokens = streams * config.num_tokens();
    let prefix = config.num_prefix_tokens();

    // Projection weights, in whichever storage the config asks for. Only
    // the large 2D matrices participate; everything else stays f32.
    let weight = |g: &mut Graph, name: &str, shape: &[usize]| {
        if config.f16_weights {
            g.parameter_f16(name, shape)
        } else {
            g.parameter(name, shape)
        }
    };

    // --- Patch embedding ---
    //
    // The reference applies a Conv2d with kernel == stride == patch_size,
    // which is exactly a per-patch linear map. Feeding pre-flattened
    // patches turns it into a single matmul and keeps the (cheap,
    // memory-bound) patch extraction outside the graph, where a render
    // pass can do it straight from the camera texture later.
    let patches = g.input(
        "patches",
        &[streams * config.num_patches(), config.patch_dim()],
    );
    let patch_w = g.parameter(
        "embeddings.patch_embeddings.weight",
        &[config.patch_dim(), hidden],
    );
    let patch_b = g.parameter("embeddings.patch_embeddings.bias", &[hidden]);
    let patch_embeds = g.matmul(patches, patch_w);
    let patch_embeds = g.bias_add(patch_embeds, patch_b);

    // --- Prepend CLS + register tokens ---
    //
    // One `[prefix, hidden]` parameter rather than separate cls_token and
    // register_tokens nodes: the loader concatenates them, which costs one
    // memcpy offline and saves a concat per forward pass.
    let prefix_tokens = g.parameter("prefix_tokens", &[prefix, hidden]);
    let prefix_tokens = stack_streams(g, &vec![prefix_tokens; streams], prefix * hidden);
    let mut x = g.concat(
        prefix_tokens,
        patch_embeds,
        streams as u32,
        (prefix * hidden) as u32,
        (config.num_patches() * hidden) as u32,
        1,
    );
    x = g.reshape(x, &[tokens, hidden]);

    // RoPE tables are the same for every layer and for both Q and K, so
    // they become two constant buffers shared by all 24 uses.
    let (cos_data, sin_data) = rope_tables(config);
    let cos = g.constant(cos_data.repeat(streams), &[tokens, hidden]);
    let sin = g.constant(sin_data.repeat(streams), &[tokens, hidden]);

    for i in 0..config.num_hidden_layers {
        let p = format!("layer.{i}");

        // --- Attention block ---
        let n1_w = g.parameter(&format!("{p}.norm1.weight"), &[hidden]);
        let n1_b = g.parameter(&format!("{p}.norm1.bias"), &[hidden]);
        let h = g.layer_norm(x, n1_w, n1_b, eps);

        let wq = weight(
            g,
            &format!("{p}.attention.q_proj.weight"),
            &[hidden, hidden],
        );
        let bq = g.parameter(&format!("{p}.attention.q_proj.bias"), &[hidden]);
        let wk = weight(
            g,
            &format!("{p}.attention.k_proj.weight"),
            &[hidden, hidden],
        );
        let wv = weight(
            g,
            &format!("{p}.attention.v_proj.weight"),
            &[hidden, hidden],
        );
        let bv = g.parameter(&format!("{p}.attention.v_proj.bias"), &[hidden]);

        let q = g.matmul(h, wq);
        let q = g.bias_add(q, bq);
        // `key_bias: false` in the config — K genuinely has no bias.
        let k = g.matmul(h, wk);
        let v = g.matmul(h, wv);
        let v = g.bias_add(v, bv);

        let q = apply_rope(g, q, cos, sin, tokens, heads, head_dim);
        let k = apply_rope(g, k, cos, sin, tokens, heads, head_dim);

        // Bidirectional within each image, never across streams.
        let attn = if streams == 1 {
            g.full_attention(q, k, v, heads, heads, head_dim)
        } else {
            let attention: Vec<_> = (0..streams)
                .map(|stream| {
                    let mut inputs = [q, k, v];
                    for input in &mut inputs {
                        let slice =
                            stream_rows(g, *input, stream, streams, config.num_tokens() * hidden);
                        *input = g.reshape(slice, &[config.num_tokens(), hidden]);
                    }
                    g.full_attention(inputs[0], inputs[1], inputs[2], heads, heads, head_dim)
                })
                .collect();
            let attn = stack_streams(g, &attention, config.num_tokens() * hidden);
            g.reshape(attn, &[tokens, hidden])
        };

        let wo = weight(
            g,
            &format!("{p}.attention.o_proj.weight"),
            &[hidden, hidden],
        );
        let bo = g.parameter(&format!("{p}.attention.o_proj.bias"), &[hidden]);
        let attn = g.matmul(attn, wo);
        let attn = g.bias_add(attn, bo);

        // LayerScale: learned gain over the trailing hidden dimension.
        let ls1 = g.parameter(&format!("{p}.layer_scale1.lambda1"), &[hidden]);
        let attn = layer_scale(g, attn, ls1, tokens, hidden);
        x = g.add(x, attn);

        // --- MLP block ---
        let n2_w = g.parameter(&format!("{p}.norm2.weight"), &[hidden]);
        let n2_b = g.parameter(&format!("{p}.norm2.bias"), &[hidden]);
        let h = g.layer_norm(x, n2_w, n2_b, eps);

        let up_w = weight(
            g,
            &format!("{p}.mlp.up_proj.weight"),
            &[hidden, config.intermediate_size],
        );
        let up_b = g.parameter(
            &format!("{p}.mlp.up_proj.bias"),
            &[config.intermediate_size],
        );
        let down_w = weight(
            g,
            &format!("{p}.mlp.down_proj.weight"),
            &[config.intermediate_size, hidden],
        );
        let down_b = g.parameter(&format!("{p}.mlp.down_proj.bias"), &[hidden]);

        let m = g.matmul(h, up_w);
        let m = g.bias_add(m, up_b);
        let m = g.gelu(m);
        let m = g.matmul(m, down_w);
        let m = g.bias_add(m, down_b);

        let ls2 = g.parameter(&format!("{p}.layer_scale2.lambda1"), &[hidden]);
        let m = layer_scale(g, m, ls2, tokens, hidden);
        x = g.add(x, m);
    }

    let n_w = g.parameter("norm.weight", &[hidden]);
    let n_b = g.parameter("norm.bias", &[hidden]);
    g.layer_norm(x, n_w, n_b, eps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vits16_shapes() {
        let c = Config::vits16();
        assert_eq!(c.grid(), 14);
        assert_eq!(c.num_patches(), 196);
        assert_eq!(c.num_prefix_tokens(), 5);
        assert_eq!(c.num_tokens(), 201);
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.patch_dim(), 768);
    }

    #[test]
    fn forward_macs_match_the_declared_matmuls() {
        assert_eq!(
            Config::vits16().with_layers(3).forward_macs(),
            1_217_878_272
        );
        assert_eq!(Config::vits16().forward_macs(), 4_698_104_832);
    }

    #[test]
    fn rope_prefix_rows_are_identity() {
        let c = Config::vits16();
        let (cos, sin) = rope_tables(&c);
        assert_eq!(cos.len(), c.num_tokens() * c.hidden_size);
        for t in 0..c.num_prefix_tokens() {
            for d in 0..c.hidden_size {
                assert_eq!(cos[t * c.hidden_size + d], 1.0);
                assert_eq!(sin[t * c.hidden_size + d], 0.0);
            }
        }
    }

    #[test]
    fn rope_replicates_across_heads_and_tiles_halves() {
        let c = Config::vits16();
        let (cos, sin) = rope_tables(&c);
        let hd = c.head_dim() as usize;
        let half = hd / 2;
        // Pick a patch token well away from the grid origin.
        let row = (c.num_prefix_tokens() + 100) * c.hidden_size;
        for d in 0..hd {
            // Every head sees the same angle...
            for h in 1..c.num_attention_heads as usize {
                assert_eq!(cos[row + d], cos[row + h * hd + d]);
                assert_eq!(sin[row + d], sin[row + h * hd + d]);
            }
            // ...and the upper half of each head repeats the lower half,
            // which is what makes rotate_half pair d with d + half.
            if d < half {
                assert_eq!(cos[row + d], cos[row + d + half]);
                assert_eq!(sin[row + d], sin[row + d + half]);
            }
        }
    }

    #[test]
    fn rope_center_of_grid_is_near_zero_angle() {
        // A 14×14 grid has no patch exactly at the center, but the two
        // straddling it must have opposite-signed angles.
        let c = Config::vits16();
        let (_, sin) = rope_tables(&c);
        let grid = c.grid();
        let hidden = c.hidden_size;
        let lo = (c.num_prefix_tokens() + (grid / 2 - 1) * grid + grid / 2 - 1) * hidden;
        let hi = (c.num_prefix_tokens() + (grid / 2) * grid + grid / 2) * hidden;
        assert!(
            sin[lo] < 0.0 && sin[hi] > 0.0,
            "expected opposite signs straddling the grid center, got {} and {}",
            sin[lo],
            sin[hi]
        );
    }

    #[test]
    fn graph_builds_and_declares_expected_slots() {
        let c = Config::vits16();
        let mut g = Graph::new();
        let out = build_encoder(&mut g, &c);
        assert_eq!(g.node(out).ty.shape, vec![c.num_tokens(), c.hidden_size]);
    }

    #[test]
    fn batched_graph_shares_parameters_and_keeps_attention_per_image() {
        use meganeura::graph::Op;

        let c = Config::vits16();
        let parameters = |g: &Graph| {
            g.nodes()
                .iter()
                .filter_map(|node| match &node.op {
                    Op::Parameter { name } => {
                        Some((name.clone(), node.ty.shape.clone(), node.ty.dtype))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let mut serial = Graph::new();
        build_encoder(&mut serial, &c);
        for streams in [1, 2, 3, 8] {
            let mut g = Graph::new();
            let out = build_batched_encoder(&mut g, &c, streams);
            assert_eq!(
                g.node(out).ty.shape,
                [streams * c.num_tokens(), c.hidden_size]
            );
            assert_eq!(parameters(&g), parameters(&serial));
            let input = g
                .nodes()
                .iter()
                .find(|node| matches!(&node.op, Op::Input { name } if name == "patches"))
                .unwrap();
            assert_eq!(input.ty.shape, [streams * c.num_patches(), c.patch_dim()]);
            let attentions: Vec<_> = g
                .nodes()
                .iter()
                .filter(|node| matches!(node.op, Op::FullAttention { .. }))
                .collect();
            assert_eq!(attentions.len(), streams * c.num_hidden_layers);
            for node in attentions {
                assert_eq!(node.ty.shape, [c.num_tokens(), c.hidden_size]);
                for &input in &node.inputs {
                    assert_eq!(g.node(input).ty.shape, [c.num_tokens(), c.hidden_size]);
                }
            }
            let (cos, sin) = rope_tables(&c);
            let tables: Vec<_> = g
                .nodes()
                .iter()
                .filter_map(|node| match &node.op {
                    Op::Constant { data } => Some(data.as_slice()),
                    _ => None,
                })
                .collect();
            assert_eq!(tables, [cos.repeat(streams), sin.repeat(streams)]);
        }
    }

    #[test]
    #[should_panic(expected = "at least one stream")]
    fn empty_batch_is_not_a_valid_encoder() {
        build_batched_encoder(&mut Graph::new(), &Config::vits16(), 0);
    }
}
