//! # Training Loop
//!
//! Sets up the data pipeline, optimizer, and Burn `Learner` then launches
//! training.  After training the model is serialized to disk so it can be
//! loaded later for inference.
//!
//! ## Burn training stack overview
//!
//! ```text
//! TrainConfig          ← hyperparameters (serialized to JSON)
//!   │
//!   ├─ UNetConfig.init()   → UNet<B>          ← the model
//!   ├─ AdamConfig.init()   → Adam optimizer
//!   ├─ DataLoaderBuilder   → train / valid DataLoader
//!   │
//!   ▼
//! SupervisedTraining
//!   ├─ metrics (LossMetric)        ← like Lightning callbacks
//!   ├─ checkpointer (CompactRecorder)
//!   └─ .launch(Learner::new(model, optimizer, lr))
//!        │
//!        ▼
//!     for each epoch / batch:
//!       model.step(batch)   ← TrainStep trait
//!       optimizer.step(grads)
//! ```
//!
//! ## Python ↔ Burn comparison
//!
//! - `Adam(model.parameters(), lr=1e-3)` → `AdamConfig::new().init()`
//! - `for epoch in range(n): for batch in loader:` →
//!   `SupervisedTraining::launch(Learner::new(...))`
//! - `loss.backward(); optimizer.step()` → handled inside `TrainStep::step()`
//! - `torch.save(model.state_dict(), path)` → `model.save_file(path,
//!   &Recorder::new())`

use crate::data::{SegmentationBatcher, with_transforms};
use crate::dataset::SyntheticEllipseDataset;
use crate::model::UNetConfig;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::{CompactRecorder, NoStdTrainingRecorder};
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::LossMetric;
use burn::train::{Learner, SupervisedTraining};

/// Top-level experiment configuration – serializable so we can save / reload
/// it alongside the model checkpoint.
///
/// **`#[derive(Config)]`** auto-generates:
/// - `TrainConfig::new(optimizer, model)` (fields without `#[config(default)]`
///   become constructor args)
/// - `.save(path)` / `TrainConfig::load(path)` for JSON serialization
/// - `Default`-like defaults for annotated fields
///
/// This is Burn's answer to experiment tracking — every run's config is
/// persisted to `config.json` inside the artifact directory.
#[derive(Config, Debug)]
pub struct TrainConfig {
    /// Number of training epochs.
    #[config(default = 5)]
    pub num_epochs: usize,
    /// Number of samples in the training set.
    #[config(default = 200)]
    pub train_size: usize,
    /// Number of samples in the validation set.
    #[config(default = 50)]
    pub valid_size: usize,
    /// Mini-batch size.
    #[config(default = 4)]
    pub batch_size: usize,
    /// Number of data-loading workers.
    #[config(default = 2)]
    pub num_workers: usize,
    /// Random seed.
    #[config(default = 42)]
    pub seed: u64,
    /// Adam optimiser configuration.
    pub optimizer: AdamConfig,
    /// Model configuration.
    pub model: UNetConfig,
}

/// Run the full training pipeline.
///
/// # Arguments
/// * `artifact_dir` – directory where checkpoints, config, and the final model
///   are written.
/// * `device` – the target backend device (e.g. CPU or GPU).
///
/// ## Backend constraint: `AutodiffBackend`
///
/// This function requires `B: AutodiffBackend`, not just `Backend`, because
/// training needs `.backward()` for gradient computation.  The `Autodiff<B>`
/// wrapper (set up in `main.rs`) fulfils this constraint.  Inference uses
/// plain `Backend` since no gradients are needed.
pub fn run<B: AutodiffBackend>(
    artifact_dir: &str,
    device: B::Device,
    config_path: Option<&str>,
) {
    // ── Create (or clean) artifact directory ──────────────────────────────
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir)
        .expect("failed to create artifact dir");

    // ── Config ────────────────────────────────────────────────────────────
    let config = match config_path {
        Some(path) => {
            println!("Loading config from {path}");
            TrainConfig::load(path)
                .expect("failed to load config")
        }
        None => {
            let optimizer = AdamConfig::new();
            let model_cfg = UNetConfig::new();
            TrainConfig::new(optimizer, model_cfg)
        }
    };

    // Persist config for reproducibility.
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("failed to save config");

    // ── Seed ──────────────────────────────────────────────────────────────
    B::seed(&device, config.seed);

    // ── Model ─────────────────────────────────────────────────────────────
    let model = config.model.init::<B>(&device);

    // ── Datasets with transforms ─────────────────────────────────────────
    let train_dataset =
        with_transforms(SyntheticEllipseDataset::train(config.train_size));
    let valid_dataset =
        with_transforms(SyntheticEllipseDataset::validation(config.valid_size));

    println!("Train dataset size: {}", train_dataset.len());
    println!("Valid dataset size: {}", valid_dataset.len());

    // ── Data loaders ─────────────────────────────────────────────────────
    let batcher_train = SegmentationBatcher::new();
    let batcher_valid = SegmentationBatcher::new();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    // ── Training ─────────────────────────────────────────────────────────
    let training = SupervisedTraining::new(
        artifact_dir,
        dataloader_train,
        dataloader_valid,
    )
    .metric_train_numeric(LossMetric::new())
    .metric_valid_numeric(LossMetric::new())
    .with_file_checkpointer(CompactRecorder::new())
    .num_epochs(config.num_epochs)
    .summary();

    let lr = 1e-3;
    let result =
        training.launch(Learner::new(model, config.optimizer.init(), lr));

    // ── Save final model ─────────────────────────────────────────────────
    result
        .model
        .save_file(
            format!("{artifact_dir}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("failed to save trained model");

    println!("Training complete – artifacts written to {artifact_dir}/");
}
