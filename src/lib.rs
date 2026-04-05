//! # learn-burn
//!
//! Educational project demonstrating how to train and run inference with a
//! **UNet-like segmentation model** using the [Burn](https://burn.dev)
//! deep-learning framework in Rust.
//!
//! ## How the pieces fit together
//!
//! The data pipeline and training loop follow the same conceptual flow you
//! know from PyTorch, but expressed through Burn's trait system:
//!
//! ```text
//! dataset.rs      data.rs          model.rs
//! ──────────      ───────          ────────
//! Dataset<Item> ─► MapperDataset ─► UNet<B>
//!  .get(i)         (transforms)     .forward_step()
//!                  Batcher─►Batch<B>
//!                                  training.rs
//!                                  ───────────
//!                           ◄───── Learner +
//!                                  SupervisedTraining
//! ```
//!
//! ## Module overview
//!
//! - [`dataset`] — *(`torch.utils.data.Dataset`)* Synthetic
//!   ellipse-segmentation generator implementing Burn's `Dataset` trait.
//! - [`data`] — *(`torchvision.transforms` + `collate_fn`)* Transforms
//!   (`Mapper`), batching (`Batcher`), and the `SegmentationBatch` struct.
//! - [`model`] — *(`nn.Module` subclass)* UNet with skip connections and aux
//!   cls/reg heads.
//! - [`training`] — *(PyTorch Lightning `Trainer`)* Training loop: `Learner` +
//!   `SupervisedTraining`.
//! - [`inference`] — *(`model.eval(); model(x)`)* Loading a saved model and
//!   running predictions.
//!
//! ## Backend selection
//!
//! Burn is **backend-agnostic**: every tensor and module is generic over a
//! `Backend` trait.  The concrete backend is chosen at compile time via Cargo
//! features — see `main.rs` and the `[features]` table in `Cargo.toml`.
//!
//! Available features: `tch-gpu` (default), `tch-cpu`, `ndarray`, `wgpu`.

pub mod data;
pub mod dataset;
pub mod inference;
pub mod model;
pub mod training;
