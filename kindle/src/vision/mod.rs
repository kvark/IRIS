//! Frozen visual perception for Dreamer.
//!
//! The encoder runs in a separate inference-only Meganeura session. Its
//! parameters therefore cannot accidentally enter the Dreamer optimizer.
//! Dreamer consumes a fixed-size projected patch grid. Encoder identity and
//! temporal semantics are checkpointed; equal shapes do not mean equal features.

use std::path::Path;
use std::sync::Arc;

use meganeura::data::safetensors::SafeTensorsModel;
use meganeura::{Graph, Mode, NodeId, Session, SessionConfig};

pub mod dinov3;
pub mod levjepa;
pub mod preprocess;
mod weights;

/// Meta's upstream DINOv3 source revision used to validate this port.
pub const DINOV3_UPSTREAM_REV: &str = "6876159a11b4df116f30f667f8c9888617df0751";
/// Source revision of the native Meganeura transcription adapted here.
pub const DINOVISION_SOURCE_REV: &str = "dc35cdf1c7c910cdd93c5b5362846842ae469a21";
/// Hugging Face model repository expected by [`DinoEncoder::load_vits16`].
pub const VITS16_MODEL_ID: &str = "facebook/dinov3-vits16-pretrain-lvd1689m";
/// Immutable Hugging Face snapshot used for the numerical golden values.
pub const VITS16_CHECKPOINT_REV: &str = "114c1379950215c8b35dfcd4e90a5c251dde0d32";
/// SHA-256/LFS object ID of the pinned reference weight file.
pub const VITS16_CHECKPOINT_SHA256: &str =
    "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d";
/// Channels retained by the fixed Johnson–Lindenstrauss projection.
pub const OBSERVATION_CHANNELS: usize = 64;
/// Spatial side after fixed 2×2 pooling of DINO's 14×14 patch grid.
pub const OBSERVATION_GRID: usize = 7;
/// Stable seed for the non-trainable projection matrix.
pub const PROJECTION_SEED: u64 = 0xd1_30_00_03_00_00_00_01;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PerceptionKind {
    DinoV3,
    LeVJepa,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PerceptionIdentity {
    pub kind: PerceptionKind,
    pub model_id: String,
    pub checkpoint_revision: String,
    pub encoding_revision: String,
    pub checkpoint_sha256: String,
}

impl PerceptionKind {
    pub fn identity(self, fingerprint: String) -> PerceptionIdentity {
        let (model_id, checkpoint_revision, encoding_revision) = match self {
            Self::DinoV3 => (
                VITS16_MODEL_ID,
                VITS16_CHECKPOINT_REV,
                "dinov3-vits16-letterbox224-jl64-pool2-v1",
            ),
            Self::LeVJepa => (
                levjepa::MODEL_ID,
                levjepa::CHECKPOINT_REV,
                levjepa::ENCODING_REV,
            ),
        };
        PerceptionIdentity {
            kind: self,
            model_id: model_id.to_owned(),
            checkpoint_revision: checkpoint_revision.to_owned(),
            encoding_revision: encoding_revision.to_owned(),
            checkpoint_sha256: fingerprint,
        }
    }
}

impl PerceptionIdentity {
    pub(crate) fn validate(&self) -> std::io::Result<()> {
        let hash = &self.checkpoint_sha256;
        if hash.len() != 64
            || !hash
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid perception fingerprint",
            ));
        }
        if *self != self.kind.identity(hash.clone()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unsupported perception identity or temporal encoding",
            ));
        }
        Ok(())
    }

    pub(crate) fn verify_file(&self, checkpoint: &Path) -> std::io::Result<()> {
        self.validate()?;
        let actual = checkpoint_sha256(checkpoint)?;
        if actual != self.checkpoint_sha256 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "perception checkpoint SHA-256 is {actual}, expected {}",
                    self.checkpoint_sha256
                ),
            ));
        }
        Ok(())
    }
}

pub(crate) enum Perception {
    Dino(DinoPerception),
    LeVJepa(levjepa::LeVJepaPerception),
}

impl Perception {
    pub(crate) fn load(
        kind: PerceptionKind,
        checkpoint: &Path,
        gpu: Arc<blade_graphics::Context>,
        cache: Option<&Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_batched(kind, checkpoint, 1, gpu, cache)
    }

    pub(crate) fn load_batched(
        kind: PerceptionKind,
        checkpoint: &Path,
        streams: usize,
        gpu: Arc<blade_graphics::Context>,
        cache: Option<&Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        match kind {
            PerceptionKind::DinoV3 => Ok(Self::Dino(DinoPerception::load_batched(
                checkpoint,
                streams,
                Some(gpu),
                cache,
            )?)),
            PerceptionKind::LeVJepa => Ok(Self::LeVJepa(levjepa::LeVJepaPerception::load_batched(
                checkpoint,
                streams,
                Some(gpu),
                cache,
            )?)),
        }
    }

    pub(crate) fn reset(&mut self) {
        if let Self::LeVJepa(encoder) = self {
            encoder.reset();
        }
    }

    pub(crate) fn encode_frame_rgb8(
        &mut self,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> Observation {
        match self {
            Self::Dino(encoder) => encoder.encode_frame_rgb8(rgb, width, height),
            Self::LeVJepa(encoder) => encoder.encode_frame_rgb8(rgb, width, height),
        }
    }

    pub(crate) fn encode_frames_rgb8(
        &mut self,
        arrivals: &[(usize, &crate::RgbFrame, bool)],
    ) -> Vec<Observation> {
        match self {
            Self::Dino(encoder) => encoder.encode_frames_rgb8(arrivals),
            Self::LeVJepa(encoder) => encoder.encode_frames_rgb8(arrivals),
        }
    }
}

pub(crate) fn checkpoint_sha256(path: &Path) -> std::io::Result<String> {
    hash_reader(std::fs::File::open(path)?)
}

fn hash_reader(mut reader: impl std::io::Read) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};

    let mut digest = Sha256::new();
    let mut buffer = [0; 65_536];
    loop {
        let count = match reader.read(&mut buffer) {
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if count == 0 {
            return Ok(format!("{:x}", digest.finalize()));
        }
        digest.update(&buffer[..count]);
    }
}

/// One frozen, compressed visual observation in token-major `[7 * 7, 64]`
/// order. Replay retains the exact features observed at collection time.
#[derive(Clone, Debug)]
pub struct Observation {
    values: Box<[f32]>,
}

impl Observation {
    pub const LEN: usize = OBSERVATION_GRID * OBSERVATION_GRID * OBSERVATION_CHANNELS;

    pub fn from_vec(values: Vec<f32>) -> Self {
        assert_eq!(values.len(), Self::LEN);
        assert!(values.iter().all(|value| value.is_finite()));
        Self {
            values: values.into_boxed_slice(),
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }
}

/// Frozen DINOv3 ViT-S/16 inference session.
pub struct DinoEncoder {
    config: dinov3::Config,
    session: Session,
    input: Vec<f32>,
    tokens: Vec<f32>,
}

/// Production perception path: full frozen ViT-S/16 followed by a frozen
/// random projection and fixed spatial pooling.
///
/// The projection is embedded as a graph constant. It has no optimizer state
/// and never changes, so cached replay observations remain valid forever.
pub struct DinoPerception {
    config: dinov3::Config,
    streams: usize,
    session: Session,
    input: Vec<f32>,
    projected: Vec<f32>,
    pooled: Vec<f32>,
}

impl DinoPerception {
    pub fn load_vits16(
        checkpoint: impl AsRef<Path>,
        gpu: Option<Arc<blade_graphics::Context>>,
        plan_cache: Option<&Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_batched(checkpoint, 1, gpu, plan_cache)
    }

    /// One frozen ViT-S/16, with batched dense layers and independent image attention.
    pub fn load_batched(
        checkpoint: impl AsRef<Path>,
        streams: usize,
        gpu: Option<Arc<blade_graphics::Context>>,
        plan_cache: Option<&Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if streams == 0 {
            return Err("DINOv3 needs at least one stream".into());
        }
        let gpu = match gpu {
            Some(gpu) => gpu,
            None => Arc::new(crate::init_gpu_context()?),
        };
        let config = dinov3::Config::vits16();
        let mut graph = Graph::new();
        let tokens = dinov3::build_batched_encoder(&mut graph, &config, streams);
        let prefix_values = config.num_prefix_tokens() * config.hidden_size;
        let patch_values = config.num_patches() * config.hidden_size;
        let patches = graph.split_b(
            tokens,
            streams as u32,
            prefix_values as u32,
            patch_values as u32,
            1,
        );
        let patches = graph.reshape(
            patches,
            &[streams * config.num_patches(), config.hidden_size],
        );
        let projection = graph.constant(
            fixed_projection(config.hidden_size, OBSERVATION_CHANNELS, PROJECTION_SEED),
            &[config.hidden_size, OBSERVATION_CHANNELS],
        );
        let projected = graph.matmul(patches, projection);
        graph.set_outputs(vec![projected]);

        let (mut session, _) = meganeura::build(
            &graph,
            SessionConfig {
                mode: Mode::Inference,
                gpu: Some(gpu),
                cache: plan_cache,
                ..SessionConfig::default()
            },
        );
        let model = SafeTensorsModel::load(checkpoint.as_ref().to_path_buf())?;
        weights::load_encoder(&mut session, &model, &config)?;

        Ok(Self {
            input: vec![0.0; streams * config.num_patches() * config.patch_dim()],
            projected: vec![0.0; streams * config.num_patches() * OBSERVATION_CHANNELS],
            pooled: vec![0.0; streams * Observation::LEN],
            streams,
            config,
            session,
        })
    }

    pub fn gpu_device(&self) -> crate::GpuDeviceInfo {
        crate::gpu_device_info(self.session.device_information())
    }

    pub fn encode_rgb8(&mut self, rgb: &[u8]) -> Observation {
        assert_eq!(self.streams, 1, "use encode_frames_rgb8 for a batch");
        self.input =
            preprocess::patches_from_rgb8(rgb, self.config.image_size, self.config.patch_size);
        self.run_batch(&[0]).pop().unwrap()
    }

    /// Encode any non-empty RGB8 frame using deterministic, aspect-preserving
    /// letterboxing to DINO's fixed 224×224 input.
    pub fn encode_frame_rgb8(&mut self, rgb: &[u8], width: usize, height: usize) -> Observation {
        let resized = preprocess::resize_letterbox_rgb8(rgb, width, height, self.config.image_size);
        self.encode_rgb8(&resized)
    }

    pub fn encode_normalized_chw(&mut self, pixels: &[f32]) -> Observation {
        assert_eq!(self.streams, 1, "use encode_frames_rgb8 for a batch");
        self.input = preprocess::patches_from_pixels_chw(
            pixels,
            self.config.image_size,
            self.config.patch_size,
        );
        self.run_batch(&[0]).pop().unwrap()
    }

    /// Encode selected arrivals in one GPU pass, returning the caller's order.
    /// DINO has no temporal state, so episode reset flags do not affect encoding.
    pub fn encode_frames_rgb8(
        &mut self,
        arrivals: &[(usize, &crate::RgbFrame, bool)],
    ) -> Vec<Observation> {
        let mut present = vec![false; self.streams];
        for &(stream, _, _) in arrivals {
            assert!(
                stream < self.streams && !present[stream],
                "invalid or repeated stream"
            );
            present[stream] = true;
        }
        if arrivals.is_empty() {
            return Vec::new();
        }
        self.input.fill(0.0);
        let width = self.config.num_patches() * self.config.patch_dim();
        for &(stream, frame, _) in arrivals {
            let rgb = preprocess::resize_letterbox_rgb8(
                frame.pixels(),
                frame.width(),
                frame.height(),
                self.config.image_size,
            );
            self.input[stream * width..(stream + 1) * width].copy_from_slice(
                &preprocess::patches_from_rgb8(
                    &rgb,
                    self.config.image_size,
                    self.config.patch_size,
                ),
            );
        }
        self.run_batch(&arrivals.iter().map(|a| a.0).collect::<Vec<_>>())
    }

    /// Stream-major projected patch tokens before the production 2x2 pooling.
    ///
    /// This is exposed for representation diagnostics. Dreamer continues to
    /// consume only the pooled [`Observation`], so reading these values
    /// cannot change training or replay compatibility.
    pub fn projected_patches(&self) -> &[f32] {
        &self.projected
    }

    /// Side length of the square grid returned by [`Self::projected_patches`].
    pub fn projected_grid(&self) -> usize {
        self.config.grid()
    }

    fn run_batch(&mut self, active: &[usize]) -> Vec<Observation> {
        self.session.set_input("patches", &self.input);
        self.session.step();
        self.session.wait();
        self.session.read_output_by_index(0, &mut self.projected);
        active
            .iter()
            .map(|&stream| {
                let width = self.config.num_patches() * OBSERVATION_CHANNELS;
                let pooled =
                    &mut self.pooled[stream * Observation::LEN..(stream + 1) * Observation::LEN];
                pool_2x2_token_major(
                    &self.projected[stream * width..(stream + 1) * width],
                    self.config.grid(),
                    OBSERVATION_CHANNELS,
                    pooled,
                );
                Observation::from_vec(pooled.to_vec())
            })
            .collect()
    }
}

fn stream_rows(
    g: &mut Graph,
    mut x: NodeId,
    stream: usize,
    streams: usize,
    width: usize,
) -> NodeId {
    if stream > 0 {
        x = g.split_b(x, 1, stream as u32, (streams - stream) as u32, width as u32);
    }
    if stream + 1 < streams {
        x = g.split_a(x, 1, 1, (streams - stream - 1) as u32, width as u32);
    }
    x
}

fn stack_streams(g: &mut Graph, rows: &[NodeId], width: usize) -> NodeId {
    let mut x = rows[0];
    for (stream, &row) in rows.iter().enumerate().skip(1) {
        x = g.concat(x, row, 1, stream as u32, 1, width as u32);
    }
    x
}

pub(crate) fn fixed_projection(input: usize, output: usize, seed: u64) -> Vec<f32> {
    assert!(input > 0 && output > 0);
    let scale = 1.0 / (output as f32).sqrt();
    let mut state = seed;
    (0..input * output)
        .map(|_| {
            // xorshift64*: deterministic across platforms and Rust versions.
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bit = state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 63;
            if bit == 0 { -scale } else { scale }
        })
        .collect()
}

fn pool_2x2_token_major(input: &[f32], grid: usize, channels: usize, output: &mut [f32]) {
    assert_eq!(grid % 2, 0);
    assert_eq!(input.len(), grid * grid * channels);
    let out_grid = grid / 2;
    assert_eq!(output.len(), out_grid * out_grid * channels);
    for y in 0..out_grid {
        for x in 0..out_grid {
            for channel in 0..channels {
                let mut sum = 0.0;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let token = (2 * y + dy) * grid + 2 * x + dx;
                        sum += input[token * channels + channel];
                    }
                }
                output[(y * out_grid + x) * channels + channel] = 0.25 * sum;
            }
        }
    }
}

impl DinoEncoder {
    /// Build the full 12-layer ViT-S/16 and load a local safetensors file.
    ///
    /// The checkpoint is deliberately external: DINOv3 weights are covered
    /// by Meta's DINOv3 license and must not be committed to this repository.
    /// `plan_cache` is optional but strongly recommended after the first run.
    pub fn load_vits16(
        checkpoint: impl AsRef<Path>,
        gpu: Option<Arc<blade_graphics::Context>>,
        plan_cache: Option<&Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load(dinov3::Config::vits16(), checkpoint, gpu, plan_cache)
    }

    /// Build a frozen encoder with an explicit DINO configuration.
    pub fn load(
        config: dinov3::Config,
        checkpoint: impl AsRef<Path>,
        gpu: Option<Arc<blade_graphics::Context>>,
        plan_cache: Option<&Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let gpu = match gpu {
            Some(gpu) => gpu,
            None => Arc::new(crate::init_gpu_context()?),
        };
        let mut graph = Graph::new();
        let output = dinov3::build_encoder(&mut graph, &config);
        graph.set_outputs(vec![output]);

        let (mut session, _) = meganeura::build(
            &graph,
            SessionConfig {
                mode: Mode::Inference,
                gpu: Some(gpu),
                cache: plan_cache,
                ..SessionConfig::default()
            },
        );
        let model = SafeTensorsModel::load(checkpoint.as_ref().to_path_buf())?;
        weights::load_encoder(&mut session, &model, &config)?;

        Ok(Self {
            input: vec![0.0; config.num_patches() * config.patch_dim()],
            tokens: vec![0.0; config.num_tokens() * config.hidden_size],
            config,
            session,
        })
    }

    pub fn gpu_device(&self) -> crate::GpuDeviceInfo {
        crate::gpu_device_info(self.session.device_information())
    }

    pub fn config(&self) -> &dinov3::Config {
        &self.config
    }

    /// Encode an exact-size interleaved RGB8 frame and return a row-major
    /// `[grid * grid, hidden]` patch-token matrix.
    ///
    /// This method does not resize. The caller must provide
    /// `image_size × image_size × 3` bytes, currently 224×224 RGB.
    pub fn encode_rgb8(&mut self, rgb: &[u8]) -> &[f32] {
        self.input =
            preprocess::patches_from_rgb8(rgb, self.config.image_size, self.config.patch_size);
        self.run()
    }

    /// Encode normalized CHW pixels using the same tensor convention as the
    /// Hugging Face reference processor.
    pub fn encode_normalized_chw(&mut self, pixels: &[f32]) -> &[f32] {
        self.input = preprocess::patches_from_pixels_chw(
            pixels,
            self.config.image_size,
            self.config.patch_size,
        );
        self.run()
    }

    /// Encode an already patchified, ImageNet-normalized frame.
    pub fn encode_patches(&mut self, patches: &[f32]) -> &[f32] {
        assert_eq!(patches.len(), self.input.len());
        self.input.copy_from_slice(patches);
        self.run()
    }

    fn run(&mut self) -> &[f32] {
        self.session.set_input("patches", &self.input);
        self.session.step();
        self.session.wait();
        self.session.read_output_by_index(0, &mut self.tokens);
        let offset = self.config.num_prefix_tokens() * self.config.hidden_size;
        &self.tokens[offset..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_matches_sha256_known_vectors_and_chunked_reads() {
        use sha2::{Digest, Sha256};

        assert_eq!(
            hash_reader(&b"abc"[..]).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hash_reader(&b""[..]).unwrap(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let bytes = vec![17; 131_079];
        assert_eq!(
            hash_reader(bytes.as_slice()).unwrap(),
            format!("{:x}", Sha256::digest(&bytes))
        );
    }

    #[test]
    fn baseline_is_full_vits16_with_dense_patch_output() {
        let config = dinov3::Config::vits16();
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.image_size, 224);
        assert_eq!(config.grid(), 14);
        assert_eq!(config.num_patches(), 196);
        assert_eq!(config.hidden_size, 384);
        assert_eq!(config.num_patches() * config.hidden_size, 75_264);
    }

    #[test]
    fn projection_and_pooling_are_fixed_and_spatial() {
        assert_eq!(
            fixed_projection(384, 64, PROJECTION_SEED),
            fixed_projection(384, 64, PROJECTION_SEED)
        );
        let input: Vec<f32> = (0..14 * 14 * 2).map(|value| value as f32).collect();
        let mut output = vec![0.0; 7 * 7 * 2];
        pool_2x2_token_major(&input, 14, 2, &mut output);
        let expected = (input[0] + input[2] + input[28] + input[30]) / 4.0;
        assert_eq!(output[0], expected);
        assert_eq!(output.len(), 98);
    }

    #[test]
    fn empty_dino_batch_is_rejected_before_loading_weights_or_gpu() {
        let error = DinoPerception::load_batched("unused", 0, None, None)
            .err()
            .unwrap();
        assert!(error.to_string().contains("at least one stream"));
    }

    #[test]
    #[ignore = "requires GPU and pinned DINOv3 weights; checks batch parity and stream isolation"]
    fn dino_batch_matches_serial_with_sparse_and_reordered_arrivals() {
        let checkpoint =
            std::env::var_os("KINDLE_DINOV3_WEIGHTS").expect("set KINDLE_DINOV3_WEIGHTS");
        assert_eq!(
            checkpoint_sha256(Path::new(&checkpoint)).unwrap(),
            VITS16_CHECKPOINT_SHA256
        );
        let frames: Vec<_> = (0..6)
            .map(|index| {
                let (width, height) = [(64, 64), (160, 210), (213, 97)][index % 3];
                crate::RgbFrame::new(
                    width,
                    height,
                    (0..width * height * 3)
                        .map(|pixel| ((pixel * 37 + index * 71) % 251) as u8)
                        .collect(),
                )
            })
            .collect();
        let expected: Vec<_> = {
            let mut serial = DinoPerception::load_vits16(&checkpoint, None, None).unwrap();
            frames
                .iter()
                .map(|frame| {
                    let pooled =
                        serial.encode_frame_rgb8(frame.pixels(), frame.width(), frame.height());
                    (pooled, serial.projected_patches().to_vec())
                })
                .collect()
        };
        let mut batch = DinoPerception::load_batched(&checkpoint, 3, None, None).unwrap();
        let mut worst = 0.0_f32;
        for selections in [
            vec![(2, 2), (0, 0), (1, 1)],
            vec![],
            vec![(1, 4)],
            vec![(2, 5), (0, 3)],
            vec![(0, 0), (2, 2), (1, 1)],
        ] {
            let arrivals: Vec<_> = selections
                .iter()
                .map(|&(stream, index)| (stream, &frames[index], index % 2 == 0))
                .collect();
            let observations = batch.encode_frames_rgb8(&arrivals);
            assert_eq!(observations.len(), selections.len());
            for ((stream, index), observation) in selections.into_iter().zip(observations) {
                let width = 196 * OBSERVATION_CHANNELS;
                for (actual, reference) in [
                    (observation.as_slice(), expected[index].0.as_slice()),
                    (
                        &batch.projected_patches()[stream * width..(stream + 1) * width],
                        expected[index].1.as_slice(),
                    ),
                ] {
                    let mut error = 0.0_f64;
                    let mut energy = 0.0_f64;
                    assert_eq!(actual.len(), reference.len());
                    for (&actual, &reference) in actual.iter().zip(reference) {
                        assert!(actual.is_finite());
                        worst = worst.max((actual - reference).abs());
                        error += f64::from(actual - reference).powi(2);
                        energy += f64::from(reference).powi(2);
                    }
                    assert!(
                        (error / energy).sqrt() < 1e-4,
                        "stream {stream} image {index}"
                    );
                }
            }
        }
        assert!(worst < 0.005, "DINO batch max absolute error {worst}");
        let baseline = batch.encode_frames_rgb8(&[
            (0, &frames[0], false),
            (1, &frames[1], false),
            (2, &frames[2], false),
        ]);
        let baseline_projected = batch.projected_patches().to_vec();
        let changed = batch.encode_frames_rgb8(&[
            (0, &frames[0], true),
            (1, &frames[4], true),
            (2, &frames[2], false),
        ]);
        for stream in [0, 2] {
            assert_eq!(
                baseline[stream].as_slice(),
                changed[stream].as_slice(),
                "another stream must not affect this image"
            );
            let width = 196 * OBSERVATION_CHANNELS;
            assert_eq!(
                &baseline_projected[stream * width..(stream + 1) * width],
                &batch.projected_patches()[stream * width..(stream + 1) * width]
            );
        }
        assert_ne!(
            baseline[1].as_slice(),
            changed[1].as_slice(),
            "the perturbed image must actually change"
        );
        eprintln!(
            "DINOv3 batched/serial maximum absolute error {worst}; unperturbed streams bit-identical"
        );
    }

    /// Full-checkpoint parity against Transformers 5.15.0 / PyTorch 2.13.
    ///
    /// Run with:
    /// `KINDLE_DINOV3_WEIGHTS=/path/model.safetensors cargo test -p kindle
    /// vision::tests::vits16_checkpoint_matches_hugging_face -- --ignored`
    #[test]
    #[ignore = "requires a GPU and the separately licensed DINOv3 checkpoint"]
    fn vits16_checkpoint_matches_hugging_face() {
        let checkpoint = std::env::var_os("KINDLE_DINOV3_WEIGHTS")
            .expect("set KINDLE_DINOV3_WEIGHTS to the ViT-S/16 safetensors file");
        assert_eq!(
            checkpoint_sha256(Path::new(&checkpoint)).unwrap(),
            VITS16_CHECKPOINT_SHA256
        );
        let mut encoder =
            DinoEncoder::load_vits16(&checkpoint, None, None).expect("load DINOv3 ViT-S/16");
        let rgb: Vec<u8> = (0..224 * 224 * 3)
            .map(|index| ((index * 37 + 13) % 251) as u8)
            .collect();
        let actual = encoder.encode_rgb8(&rgb);

        // Dense patch tokens from facebook/dinov3-vits16-pretrain-lvd1689m
        // snapshot 114c137..., with the prefix tokens removed.
        let expected = [
            (0, -0.057_838_768_f32),
            (1, -0.493_787_35),
            (2, -0.066_303_96),
            (3, 0.386_416_6),
            (7, 0.100_504_76),
            (31, 0.167_517_4),
            (127, -0.292_504_4),
            (383, -0.389_186_32),
            (384, 0.038_286_045),
            (385, -0.489_355_62),
            (767, -0.390_894_17),
            (1_023, 0.072_687_8),
            (4_095, -0.063_992_37),
            (8_191, -0.234_296_11),
            (16_383, 0.103_825_44),
            (32_767, -0.341_246_3),
            (49_151, -0.098_256_67),
            (65_535, 0.098_528_79),
            (75_263, -0.465_510_22),
        ];
        let worst = expected
            .iter()
            .map(|&(index, reference)| (actual[index] - reference).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst < 0.01, "DINOv3 sample max error {worst}");

        let mean = actual.iter().sum::<f32>() / actual.len() as f32;
        let norm = actual.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((mean - -0.003_444_944).abs() < 5e-4, "mean {mean}");
        assert!((norm - 116.004_425).abs() < 0.5, "L2 norm {norm}");

        // Verify the production projection graph and CPU pooling boundary
        // against a direct projection of the already validated patch tokens.
        let patches = actual.to_vec();
        let projection = fixed_projection(384, OBSERVATION_CHANNELS, PROJECTION_SEED);
        let mut projected = vec![0.0; 196 * OBSERVATION_CHANNELS];
        for patch in 0..196 {
            for output in 0..OBSERVATION_CHANNELS {
                projected[patch * OBSERVATION_CHANNELS + output] = (0..384)
                    .map(|input| {
                        patches[patch * 384 + input]
                            * projection[input * OBSERVATION_CHANNELS + output]
                    })
                    .sum();
            }
        }
        let mut expected_observation = vec![0.0; Observation::LEN];
        pool_2x2_token_major(
            &projected,
            14,
            OBSERVATION_CHANNELS,
            &mut expected_observation,
        );
        drop(encoder);
        let mut perception = DinoPerception::load_vits16(checkpoint, None, None)
            .expect("load production DINO perception");
        let observation = perception.encode_rgb8(&rgb);
        let projection_error = observation
            .as_slice()
            .iter()
            .zip(expected_observation)
            .map(|(value, expected)| (*value - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            projection_error < 0.01,
            "production DINO projection max error {projection_error}"
        );
    }
}
