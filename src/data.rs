//! # Data Transforms & Batching
//!
//! This module defines:
//!
//! * **Transforms** – small composable operations applied to each
//!   [`EllipseItem`] before batching.  Two illustrative transforms are
//!   provided:
//!   - [`NormalizeTransform`]: scales pixel intensities from [0, 255] to [0,
//!     1].
//!   - [`FlipHorizontalTransform`]: randomly mirrors the image and mask
//!     horizontally (a common data-augmentation trick).
//!
//! * **[`SegmentationBatcher`]** – implements Burn's [`Batcher`] trait to
//!   convert a `Vec<EllipseItem>` into GPU-ready tensors grouped in a
//!   [`SegmentationBatch`].
//!
//! ## Pipeline analogy for Python devs
//!
//! ```text
//! Python / PyTorch              Burn / Rust
//! ─────────────────             ──────────────
//! transforms.Compose([          MapperDataset::new(
//!   transforms.Normalize(...),    dataset, NormalizeTransform)
//!   transforms.RandomHFlip(),   MapperDataset::new(
//! ])                              dataset, FlipHorizontalTransform)
//!
//! DataLoader(collate_fn=...)    DataLoaderBuilder::new(batcher)
//! ```
//!
//! Transforms are wired through Burn's [`MapperDataset`] so the pipeline is:
//! `SyntheticEllipseDataset → MapperDataset<NormalizeTransform> →
//! MapperDataset<FlipHorizontalTransform> → DataLoader → Batcher`.
//!
//! ## The `Mapper` trait
//!
//! `Mapper<In, Out>` is Burn's equivalent of a single transform in
//! `torchvision.transforms`.  It has one method:
//!
//! ```text
//! fn map(&self, item: &In) -> Out
//! ```
//!
//! Note that `map` takes `&self` (immutable borrow) — the transform itself
//! cannot carry mutable state.  This is intentional: it ensures transforms
//! are deterministic and thread-safe for multi-worker loading.  If you need
//! randomness, derive it from the item content (as `FlipHorizontalTransform`
//! does) or from the index.

use crate::dataset::{EllipseItem, FIXED_H, FIXED_W};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::data::dataset::transform::{Mapper, MapperDataset};
use burn::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Transform 1: Normalize pixel intensities to [0, 1]
// ─────────────────────────────────────────────────────────────────────────────

/// Scales image values from [0, 255] to [0, 1].
///
/// The mask is already in {0, 1} so it is left unchanged.
///
/// **Rust pattern — `..item.clone()`:** the `..` syntax is Rust's "struct
/// update" (like `{**item, image: new_image}` in Python).  We clone the
/// original item then overwrite just the `image` field.
#[derive(Clone, Debug)]
pub struct NormalizeTransform;

impl Mapper<EllipseItem, EllipseItem> for NormalizeTransform {
    fn map(&self, item: &EllipseItem) -> EllipseItem {
        let image: Vec<f32> = item.image.iter().map(|&v| v / 255.0).collect();
        EllipseItem {
            image,
            ..item.clone()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Transform 2: Random horizontal flip
// ─────────────────────────────────────────────────────────────────────────────

/// Randomly flips the image and mask horizontally with 50 % probability.
///
/// This is a very common data-augmentation transform in computer vision.
///
/// **Design constraint:** `Mapper::map` receives `&self` (no mutable state)
/// and `&EllipseItem` (immutable reference), so we can't use a normal RNG.
/// Instead we derive a deterministic coin-flip from a hash of the first few
/// pixel values.  This keeps the transform **pure** (same input → same
/// output) and thread-safe for multi-worker loading.
///
/// In PyTorch's `RandomHorizontalFlip` the randomness comes from
/// `torch.rand()`, which works because Python's GIL serializes access.
/// Rust has no GIL, so we need this workaround.
#[derive(Clone, Debug)]
pub struct FlipHorizontalTransform;

impl FlipHorizontalTransform {
    /// Flip a CHW buffer horizontally.
    fn flip_chw(data: &[f32], channels: usize, h: usize, w: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; data.len()];
        for c in 0..channels {
            for y in 0..h {
                for x in 0..w {
                    out[c * h * w + y * w + x] =
                        data[c * h * w + y * w + (w - 1 - x)];
                }
            }
        }
        out
    }
}

impl Mapper<EllipseItem, EllipseItem> for FlipHorizontalTransform {
    fn map(&self, item: &EllipseItem) -> EllipseItem {
        // Derive a deterministic coin flip from the first 8 pixel values.
        let hash: u32 = item
            .image
            .iter()
            .take(8)
            .enumerate()
            .map(|(i, &v)| (v as u32).wrapping_mul(i as u32 + 1))
            .sum();

        if hash % 2 == 0 {
            // No flip – return the item unchanged.
            return item.clone();
        }

        let image = Self::flip_chw(&item.image, 3, FIXED_H, FIXED_W);
        let mask = Self::flip_chw(&item.mask, 1, FIXED_H, FIXED_W);
        EllipseItem {
            image,
            mask,
            ..item.clone()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: wire transforms onto an arbitrary Dataset<EllipseItem>
// ─────────────────────────────────────────────────────────────────────────────

/// Convenience function that wraps an existing dataset with both transforms.
///
/// The returned type is fully erased behind `Box<dyn Dataset<EllipseItem>>`
/// so callers don't need to spell out the nested `MapperDataset` types.
///
/// **Rust aside — `Box<dyn Trait>`:** this is Rust's version of dynamic
/// dispatch (like a Python abstract base class reference).  The concrete
/// `MapperDataset<MapperDataset<D, ...>, ...>` type is hidden behind the
/// trait object.  The `'static` bound means the dataset can't borrow local
/// data — it must own everything.  This is required because the
/// `DataLoaderBuilder` may move the dataset to another thread.
pub fn with_transforms(
    dataset: impl Dataset<EllipseItem> + 'static,
) -> Box<dyn Dataset<EllipseItem>> {
    let normalized = MapperDataset::new(dataset, NormalizeTransform);
    let augmented = MapperDataset::new(normalized, FlipHorizontalTransform);
    Box::new(augmented)
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch type
// ─────────────────────────────────────────────────────────────────────────────

/// A mini-batch of segmentation samples, ready for the model.
///
/// Tensor shapes (B = batch size):
/// * `images`:            `[B, 3,  FIXED_H, FIXED_W]`
/// * `masks`:             `[B, 1,  FIXED_H, FIXED_W]`
/// * `binary_targets`:    `[B]`  (i32)
/// * `regression_targets`: `[B]` (f32)
///
/// **Burn generics note:** `<B: Backend>` means this struct is parameterized
/// over a backend — the exact same struct works for CPU tensors, GPU tensors,
/// or any future backend.  In PyTorch you'd just use `torch.Tensor` and
/// call `.to(device)` at runtime; here the device/backend is baked in at
/// compile time.
#[derive(Clone, Debug)]
pub struct SegmentationBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub masks: Tensor<B, 4>,
    pub binary_targets: Tensor<B, 1, Int>,
    pub regression_targets: Tensor<B, 1>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Batcher
// ─────────────────────────────────────────────────────────────────────────────

/// Converts a `Vec<EllipseItem>` into a [`SegmentationBatch`] on the target
/// device.
///
/// This is the Rust/Burn equivalent of PyTorch's `collate_fn`.  The strategy
/// is: concatenate all flat `Vec<f32>` buffers into one big `Vec`, then
/// create a single tensor and reshape it.  This is faster than creating one
/// tensor per item and stacking them (which would issue many small GPU
/// allocations).
///
/// **Trait signature explained:**
/// ```text
/// impl<B: Backend> Batcher<B, EllipseItem, SegmentationBatch<B>>
///              ^           ^  ^             ^
///              |           |  input item    output batch
///              |           backend
///              this impl is generic over any backend
/// ```
#[derive(Clone, Debug)]
pub struct SegmentationBatcher;

impl SegmentationBatcher {
    pub fn new() -> Self {
        Self
    }
}

impl<B: Backend> Batcher<B, EllipseItem, SegmentationBatch<B>>
    for SegmentationBatcher
{
    fn batch(
        &self,
        items: Vec<EllipseItem>,
        device: &B::Device,
    ) -> SegmentationBatch<B> {
        let batch_size = items.len();

        // Pre-allocate and fill flat buffers, then create tensors in one shot.
        let img_numel = 3 * FIXED_H * FIXED_W;
        let mask_numel = 1 * FIXED_H * FIXED_W;

        let mut img_buf = Vec::with_capacity(batch_size * img_numel);
        let mut mask_buf = Vec::with_capacity(batch_size * mask_numel);
        let mut binary_buf = Vec::with_capacity(batch_size);
        let mut regression_buf = Vec::with_capacity(batch_size);

        for item in &items {
            img_buf.extend_from_slice(&item.image);
            mask_buf.extend_from_slice(&item.mask);
            binary_buf.push(item.binary_target);
            regression_buf.push(item.regression_target);
        }

        let images = Tensor::<B, 1>::from_floats(img_buf.as_slice(), device)
            .reshape([batch_size, 3, FIXED_H, FIXED_W]);
        let masks = Tensor::<B, 1>::from_floats(mask_buf.as_slice(), device)
            .reshape([batch_size, 1, FIXED_H, FIXED_W]);
        let binary_targets =
            Tensor::<B, 1, Int>::from_ints(binary_buf.as_slice(), device);
        let regression_targets =
            Tensor::<B, 1>::from_floats(regression_buf.as_slice(), device);

        SegmentationBatch {
            images,
            masks,
            binary_targets,
            regression_targets,
        }
    }
}
