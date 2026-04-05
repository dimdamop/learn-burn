//! # Inference
//!
//! Loads a trained UNet model from disk and runs it on a few synthetic
//! samples, printing per-sample predictions.
//!
//! ## How model loading works in Burn
//!
//! 1. **Create a fresh model** from its config:
//!    `UNetConfig::new().init(device)`. This allocates all parameters with
//!    their initial (random) values.
//! 2. **Load the saved weights** into a `Record` via a `Recorder`:
//!    `NoStdTrainingRecorder::new().load(path, device)`.
//! 3. **Apply the record** to the model: `model.load_record(record)`. This
//!    overwrites the random weights with the trained ones.
//!
//! This is conceptually identical to PyTorch's:
//! ```python
//! model = UNet()
//! model.load_state_dict(torch.load("model.pt"))
//! ```
//!
//! Burn enforces that the record's structure matches the model's at compile
//! time (via generic types), so you can't accidentally load weights from a
//! different architecture — it simply won't compile.

use crate::data::{SegmentationBatcher, with_transforms};
use crate::dataset::SyntheticEllipseDataset;
use crate::model::{UNet, UNetConfig};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::*;
use burn::record::{NoStdTrainingRecorder, Recorder};

/// Run inference on a handful of test samples.
///
/// # Arguments
/// * `artifact_dir` – directory that contains `model.bin` and `config.json`.
/// * `device` – the backend device to run on.
/// * `num_samples` – how many test samples to run.
pub fn run<B: Backend>(
    artifact_dir: &str,
    device: B::Device,
    num_samples: usize,
) {
    // ── Load config & model ──────────────────────────────────────────────
    let record = NoStdTrainingRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("failed to load model record");

    let model: UNet<B> = UNetConfig::new().init(&device).load_record(record);

    // ── Build a small test dataset with transforms ───────────────────────
    let dataset = with_transforms(SyntheticEllipseDataset::test(num_samples));

    let batcher = SegmentationBatcher::new();

    println!("Running inference on {num_samples} test samples…\n");

    for i in 0..dataset.len().min(num_samples) {
        let item = dataset.get(i).expect("missing item");
        let gt_binary = item.binary_target;
        let gt_reg = item.regression_target;

        // Batch of size 1.
        let batch = batcher.batch(vec![item], &device);

        let output = model.forward(batch.images);

        // Segmentation: fraction of pixels predicted as foreground.
        let seg_prob = burn::tensor::activation::sigmoid(output.seg_logits);
        let fg_fraction: f32 = seg_prob.mean().into_scalar().elem();

        // Classification: sigmoid of logit.
        let cls_prob: f32 =
            burn::tensor::activation::sigmoid(output.cls_logits)
                .into_scalar()
                .elem();
        let cls_pred = if cls_prob >= 0.5 { 1 } else { 0 };

        // Regression.
        let reg_pred: f32 = output.reg_preds.into_scalar().elem();

        println!(
            "Sample {i:>3} | fg={fg_fraction:.3} \
             cls={cls_pred}(gt={gt_binary},p={cls_prob:.3}) \
             reg={reg_pred:.2}(gt={gt_reg:.2})"
        );
    }
}
