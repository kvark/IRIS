//! Same-queue handoffs between session-owned, pinned buffers.

use std::sync::Arc;

use meganeura::{Session, runtime::ExternalSlot};

pub(crate) struct DeviceCopy<'a> {
    pub source: (&'a Session, ExternalSlot<'a>),
    pub target: (&'a Session, &'a str),
    pub target_offset_bytes: usize,
}

pub(crate) struct DeviceCopies {
    gpu: Arc<blade_graphics::Context>,
    encoder: blade_graphics::CommandEncoder,
    completion: Option<blade_graphics::SyncPoint>,
}

impl DeviceCopies {
    pub fn new(gpu: Arc<blade_graphics::Context>) -> Self {
        let encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
            name: "kindle_device_copy",
            buffer_count: 1,
            manual_barriers: false,
        });
        Self {
            gpu,
            encoder,
            completion: None,
        }
    }

    /// Submit after producers and before consumers on the shared queue. The
    /// caller must complete these copies before overwriting inputs from the CPU
    /// or destroying sessions. No host wait is needed before a consumer step.
    pub fn copy(&mut self, copies: &[DeviceCopy<'_>]) {
        let regions: Vec<_> = copies
            .iter()
            .map(|copy| {
                let (source, slot) = copy.source;
                let (target, input) = copy.target;
                assert!(Arc::ptr_eq(&self.gpu, &source.context()));
                assert!(Arc::ptr_eq(&self.gpu, &target.context()));
                assert!(!std::ptr::eq(source, target), "cross-session copies only");
                let bytes = source.slot_size(slot).expect("known source slot");
                let available = target
                    .slot_size(ExternalSlot::Input(input))
                    .expect("known target input");
                let end = copy
                    .target_offset_bytes
                    .checked_add(bytes)
                    .expect("device copy size overflow");
                assert!(end <= available, "device copy exceeds target input");
                assert_eq!(copy.target_offset_bytes % size_of::<f32>(), 0);
                let source_buffer = match slot {
                    ExternalSlot::Input(name) => source.input_buffer(name),
                    ExternalSlot::Output(index) => source.output_buffer(index),
                    ExternalSlot::Parameter(_) => panic!("copy inputs or outputs only"),
                }
                .expect("known source buffer");
                let mut target_buffer = target.input_buffer(input).expect("known target buffer");
                target_buffer.offset = target_buffer
                    .offset
                    .checked_add(copy.target_offset_bytes as u64)
                    .expect("device copy offset overflow");
                (source_buffer, target_buffer, bytes)
            })
            .collect();
        if regions.is_empty() {
            return;
        }
        self.wait();
        self.encoder.start();
        {
            let mut transfer = self.encoder.transfer("kindle_device_copy");
            for (source, target, bytes) in regions {
                transfer.copy_buffer_to_buffer(source, target, bytes as u64);
            }
        }
        self.completion = Some(self.gpu.submit(&mut self.encoder));
    }

    fn wait(&mut self) {
        if let Some(completion) = self.completion.take() {
            assert!(
                self.gpu
                    .wait_for(&completion, !0)
                    .expect("GPU device copy wait failed"),
                "device copy did not complete"
            );
        }
    }
}

impl Drop for DeviceCopies {
    fn drop(&mut self) {
        self.wait();
        self.gpu.destroy_command_encoder(&mut self.encoder);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dreamer::{readback::Readback, runtime::build_session};

    #[test]
    #[ignore = "requires a GPU"]
    fn device_copies_pack_inputs_and_outputs_before_consumption() {
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let mut graph = meganeura::Graph::new();
        let input = graph.input("input", &[2, 3]);
        let negative = graph.neg(input);
        graph.set_outputs(vec![negative]);
        let mut producer = build_session(&graph, &gpu, meganeura::Mode::Inference, false);
        let mut graph = meganeura::Graph::new();
        let input = graph.input("packed", &[4, 6]);
        let negative = graph.neg(input);
        graph.set_outputs(vec![negative]);
        let mut consumer = build_session(&graph, &gpu, meganeura::Mode::Inference, false);
        let mut copies = DeviceCopies::new(Arc::clone(&gpu));
        let mut readback = Readback::new(gpu);
        consumer.set_input("packed", &[7.0; 24]);
        for scale in [1.0, 2.0, 3.0] {
            let values = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0].map(|v| scale * v);
            producer.set_input("input", &values);
            producer.step();
            copies.copy(&[
                DeviceCopy {
                    source: (&producer, ExternalSlot::Output(0)),
                    target: (&consumer, "packed"),
                    target_offset_bytes: 6 * size_of::<f32>(),
                },
                DeviceCopy {
                    source: (&producer, ExternalSlot::Input("input")),
                    target: (&consumer, "packed"),
                    target_offset_bytes: 12 * size_of::<f32>(),
                },
            ]);
            consumer.step();
            let mut actual = [0.0; 24];
            readback.read(&consumer, &mut [(0, &mut actual)]);
            assert_eq!(actual[..6], [-7.0; 6]);
            assert_eq!(actual[6..12], values);
            assert_eq!(actual[12..18], values.map(|v| -v));
            assert_eq!(actual[18..], [-7.0; 6]);
        }
        for offset in [1, 19 * size_of::<f32>(), usize::MAX] {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                copies.copy(&[DeviceCopy {
                    source: (&producer, ExternalSlot::Output(0)),
                    target: (&consumer, "packed"),
                    target_offset_bytes: offset,
                }]);
            }));
            assert!(result.is_err(), "invalid copy offset {offset} accepted");
        }
    }
}
