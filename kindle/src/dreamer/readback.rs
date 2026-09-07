//! Batched output transfers into CPU-cached memory on the shared GPU queue.

use std::sync::Arc;
use std::time::Instant;

use meganeura::{Session, runtime::ExternalSlot};

/// Host wall time at explicit readback boundaries, not GPU idle or kernel time.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct ReadbackTiming {
    pub calls: usize,
    pub bytes: usize,
    /// Validation, buffer growth and transfer-command recording.
    pub prepare_seconds: f64,
    pub submit_seconds: f64,
    /// Includes unfinished producer compute and the transfer itself.
    pub wait_seconds: f64,
    /// Copy from the completed CPU-visible buffer into output slices.
    pub copy_seconds: f64,
}

pub(crate) struct Readback {
    gpu: Arc<blade_graphics::Context>,
    encoder: blade_graphics::CommandEncoder,
    buffer: Option<blade_graphics::Buffer>,
    capacity: usize,
    timing: ReadbackTiming,
}

impl Readback {
    pub fn new(gpu: Arc<blade_graphics::Context>) -> Self {
        let encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
            name: "kindle_readback",
            buffer_count: 1,
            manual_barriers: false,
        });
        Self {
            gpu,
            encoder,
            buffer: None,
            capacity: 0,
            timing: ReadbackTiming::default(),
        }
    }

    pub fn take_timing(&mut self) -> ReadbackTiming {
        std::mem::take(&mut self.timing)
    }

    /// Kindle graphs expose f32 outputs. Copy requested prefixes in one submit;
    /// waiting on this transfer also completes the producer's earlier submit.
    pub fn read(&mut self, session: &Session, outputs: &mut [(usize, &mut [f32])]) {
        let mut sources: Vec<_> = outputs
            .iter_mut()
            .map(|(index, output)| (session, *index, &mut **output))
            .collect();
        self.read_many(&mut sources);
    }

    /// Read independent, already-submitted producers with one transfer/wait.
    pub fn read_many(&mut self, outputs: &mut [(&Session, usize, &mut [f32])]) {
        let stage = Instant::now();
        let mut bytes = 0usize;
        for (session, index, output) in outputs.iter() {
            assert!(Arc::ptr_eq(&self.gpu, &session.context()));
            let size = std::mem::size_of_val(*output);
            let available = session
                .slot_size(ExternalSlot::Output(*index))
                .expect("known output slot");
            assert!(size <= available, "readback exceeds output size");
            bytes = bytes.checked_add(size).expect("readback size overflow");
        }
        if bytes == 0 {
            return;
        }
        if bytes > self.capacity {
            if let Some(buffer) = self.buffer.take() {
                self.gpu.destroy_buffer(buffer);
            }
            self.buffer = Some(self.gpu.create_buffer(blade_graphics::BufferDesc {
                name: "kindle_readback",
                size: bytes as u64,
                memory: blade_graphics::Memory::Download,
            }));
            self.capacity = bytes;
        }
        let buffer = self.buffer.expect("allocated readback buffer");
        self.encoder.start();
        let mut offset = 0;
        {
            let mut transfer = self.encoder.transfer("kindle_readback");
            for (session, index, output) in outputs.iter() {
                let size = std::mem::size_of_val(*output);
                if size > 0 {
                    transfer.copy_buffer_to_buffer(
                        session.output_buffer(*index).expect("known output slot"),
                        buffer.at(offset as u64),
                        size as u64,
                    );
                }
                offset += size;
            }
        }
        self.timing.prepare_seconds += stage.elapsed().as_secs_f64();
        let stage = Instant::now();
        let completion = self.gpu.submit(&mut self.encoder);
        self.timing.submit_seconds += stage.elapsed().as_secs_f64();
        let stage = Instant::now();
        assert!(
            self.gpu
                .wait_for(&completion, !0)
                .expect("GPU readback wait failed"),
            "readback did not complete"
        );
        self.timing.wait_seconds += stage.elapsed().as_secs_f64();
        let stage = Instant::now();
        let mut offset = 0;
        for (_, _, output) in outputs {
            // The completed transfer initialized these aligned f32 regions in
            // CPU-visible memory. Neither allocation can be freed or mutated
            // while this method holds their exclusive borrows.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buffer.data().add(offset).cast::<f32>(),
                    output.as_mut_ptr(),
                    output.len(),
                );
            }
            offset += std::mem::size_of_val(*output);
        }
        self.timing.copy_seconds += stage.elapsed().as_secs_f64();
        self.timing.calls += 1;
        self.timing.bytes += bytes;
    }
}

impl Drop for Readback {
    fn drop(&mut self) {
        self.gpu.destroy_command_encoder(&mut self.encoder);
        if let Some(buffer) = self.buffer.take() {
            self.gpu.destroy_buffer(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readback_timing_serializes_only_numeric_fields() {
        let value = serde_json::to_value(ReadbackTiming::default()).unwrap();
        let fields = value.as_object().unwrap();
        assert_eq!(fields.len(), 6);
        assert!(fields.values().all(|value| value.as_f64() == Some(0.0)));
    }

    #[test]
    #[ignore = "requires a GPU"]
    fn tiny_readback_matches_outputs_and_reuses_storage() {
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let mut graph = meganeura::Graph::new();
        let input = graph.input("input", &[2, 3]);
        let negative = graph.neg(input);
        let squared = graph.mul(input, input);
        graph.set_outputs(vec![negative, squared]);
        let mut session =
            super::super::runtime::build_session(&graph, &gpu, meganeura::Mode::Inference, false);
        let mut readback = Readback::new(gpu);
        for scale in [1.0, 2.0] {
            let values = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0].map(|value| scale * value);
            session.set_input("input", &values);
            session.step();
            let mut actual_negative = [0.0; 6];
            let mut actual_squared = [0.0; 6];
            readback.read(
                &session,
                &mut [(1, &mut actual_squared), (0, &mut actual_negative)],
            );
            assert_eq!(actual_negative, values.map(|value| -value));
            assert_eq!(actual_squared, values.map(|value| value * value));
            let mut prefix = [0.0; 2];
            readback.read(&session, &mut [(0, &mut prefix)]);
            assert_eq!(prefix, [-values[0], -values[1]]);
            assert_eq!(readback.capacity, 12 * std::mem::size_of::<f32>());
            let timing = readback.take_timing();
            assert_eq!(timing.calls, 2);
            assert_eq!(timing.bytes, 14 * std::mem::size_of::<f32>());
            assert!(timing.prepare_seconds >= 0.0);
            assert!(timing.submit_seconds >= 0.0);
            assert!(timing.wait_seconds >= 0.0);
            assert!(timing.copy_seconds >= 0.0);
            assert_eq!(readback.take_timing().calls, 0);
            assert_eq!(readback.take_timing().bytes, 0);
            let mut empty = [];
            readback.read(&session, &mut [(0, &mut empty)]);
            assert_eq!(readback.take_timing().calls, 0);
        }
    }

    #[test]
    #[ignore = "requires a GPU"]
    fn independent_producers_share_one_readback_without_a_host_wait_between_them() {
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let mut graph = meganeura::Graph::new();
        let input = graph.input("input", &[2, 3]);
        let negative = graph.neg(input);
        graph.set_outputs(vec![negative]);
        let mut a =
            super::super::runtime::build_session(&graph, &gpu, meganeura::Mode::Inference, false);
        let mut b =
            super::super::runtime::build_session(&graph, &gpu, meganeura::Mode::Inference, false);
        let mut readback = Readback::new(gpu);
        for scale in [1.0, 2.0, 3.0] {
            a.set_input("input", &[scale; 6]);
            a.step();
            b.set_input("input", &[-scale; 6]);
            b.step();
            let mut left = [0.0; 6];
            let mut right = [0.0; 6];
            readback.read_many(&mut [(&b, 0, &mut right), (&a, 0, &mut left)]);
            assert_eq!(left, [-scale; 6]);
            assert_eq!(right, [scale; 6]);
            let timing = readback.take_timing();
            assert_eq!(timing.calls, 1);
            assert_eq!(timing.bytes, 12 * std::mem::size_of::<f32>());
        }
    }
}
