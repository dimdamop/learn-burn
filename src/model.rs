//! # UNet-like Segmentation Model
//!
//! A simplified UNet architecture for binary segmentation, implemented entirely
//! with Burn primitives.
//!
//! ## Key Burn concepts in this module
//!
//! - **`#[derive(Module)]`**: Burn's equivalent of `nn.Module` in PyTorch. The
//!   derive macro auto-generates parameter collection, device transfer, and
//!   serialization.  Every field that is itself a `Module` (e.g. `Conv2d`,
//!   `BatchNorm`) is automatically registered as a sub-module (no need for
//!   `self.add_module()` like in PyTorch).
//!
//! - **`#[derive(Config)]`**: A companion struct that stores hyperparameters
//!   (channel counts, kernel sizes, etc.) and has an `init()` method that
//!   creates the corresponding `Module`.  Think of it as a typed, serializable
//!   `dict` of constructor arguments.
//!
//! - **`<B: Backend>`**: The module is generic over a backend.  You write the
//!   architecture once; the same code runs on CPU (NdArray), GPU
//!   (LibTorch/CUDA), or WebGPU.  In PyTorch you'd call `.to(device)` at
//!   runtime; here the backend is baked in at compile time via
//!   monomorphization.
//!
//! ## Architecture overview
//!
//! ```text
//!  Input [B,3,H,W]
//!    │
//!    ▼
//!  ┌──────────────┐
//!  │ Encoder Blk 1│──► skip₁ [B,32,H,W]
//!  └──────┬───────┘
//!         │ MaxPool 2×2
//!  ┌──────▼───────┐
//!  │ Encoder Blk 2│──► skip₂ [B,64,H/2,W/2]
//!  └──────┬───────┘
//!         │ MaxPool 2×2
//!  ┌──────▼───────┐
//!  │  Bottleneck  │       [B,128,H/4,W/4]
//!  └──────┬───────┘
//!         │ Upsample 2×
//!  ┌──────▼───────┐
//!  │ Decoder Blk 2│◄── concat skip₂
//!  └──────┬───────┘
//!         │ Upsample 2×
//!  ┌──────▼───────┐
//!  │ Decoder Blk 1│◄── concat skip₁
//!  └──────┬───────┘
//!         │
//!  ┌──────▼───────┐
//!  │  1×1 Conv    │  → [B,1,H,W]  segmentation logits
//!  └──────────────┘
//! ```
//!
//! Each "block" is `Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU`.

use burn::nn::{
    BatchNorm, BatchNormConfig, PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
    pool::{
        AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig,
    },
};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{RegressionOutput, TrainOutput, TrainStep};

use crate::data::SegmentationBatch;

// ─────────────────────────────────────────────────────────────────────────────
// Convolution block: Conv → BN → ReLU → Conv → BN → ReLU
// ─────────────────────────────────────────────────────────────────────────────

/// A double-convolution block used in every stage of the UNet.
///
/// **`#[derive(Module)]` explained:** this proc macro inspects every field
/// and, if the field's type also implements `Module`, registers it as a
/// sub-module.  This gives you:
/// - Automatic parameter enumeration (like `.parameters()` in PyTorch)
/// - Recursive `.to(device)` transfer
/// - Serialization / deserialization with `save_file` / `load_record`
///
/// You do *not* write `forward()` as part of a trait — it's just a regular
/// method.  Burn doesn't have a `__call__` / `Module.forward()` protocol.
#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
}

/// Configuration for [`ConvBlock`].
///
/// **Pattern: Config → Module.**  In Burn, the convention is:
/// 1. Define a `FooConfig` with hyperparameters (channel counts, etc.).
/// 2. Implement `FooConfig::init(device) -> Foo<B>` to build the module.
///
/// This separation means you can serialize / deserialize the config (JSON)
/// independently of the learned weights, which is handy for experiment
/// tracking and reproducibility.
#[derive(Config, Debug)]
pub struct ConvBlockConfig {
    pub in_channels: usize,
    pub out_channels: usize,
}

impl ConvBlockConfig {
    /// Initialise a [`ConvBlock`] on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvBlock<B> {
        let conv1 =
            Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device);
        let bn1 = BatchNormConfig::new(self.out_channels).init(device);
        let conv2 =
            Conv2dConfig::new([self.out_channels, self.out_channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device);
        let bn2 = BatchNormConfig::new(self.out_channels).init(device);
        ConvBlock {
            conv1,
            bn1,
            conv2,
            bn2,
        }
    }
}

impl<B: Backend> ConvBlock<B> {
    /// Forward pass: `Conv → BN → ReLU → Conv → BN → ReLU`.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        burn::tensor::activation::relu(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// UNet model
// ─────────────────────────────────────────────────────────────────────────────

/// A small UNet-like model for binary segmentation.
///
/// It has two encoder stages, a bottleneck, and two decoder stages with skip
/// connections.  A final 1×1 convolution produces a single-channel output
/// (logits).
///
/// **Skip connections** are the hallmark of UNet: the encoder features at
/// each resolution are concatenated with the decoder features at the
/// matching resolution.  This helps the decoder recover fine spatial detail
/// that would otherwise be lost after downsampling.
///
/// **Auxiliary heads:** besides the main segmentation output, we attach a
/// classification head and a regression head to the bottleneck features.
/// These are trained jointly and serve as a teaching example of multi-task
/// learning in Burn.
#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    // ── Encoder ──
    enc1: ConvBlock<B>,
    pool1: MaxPool2d,
    enc2: ConvBlock<B>,
    pool2: MaxPool2d,

    // ── Bottleneck ──
    bottleneck: ConvBlock<B>,

    // ── Decoder ──
    /// 1×1 conv to reduce channels after concatenation with skip₂.
    up_conv2: Conv2d<B>,
    dec2: ConvBlock<B>,
    /// 1×1 conv to reduce channels after concatenation with skip₁.
    up_conv1: Conv2d<B>,
    dec1: ConvBlock<B>,

    // ── Segmentation head ──
    seg_head: Conv2d<B>,

    // ── Auxiliary classification / regression head (operates on bottleneck)
    // ──
    aux_pool: AdaptiveAvgPool2d,
    aux_cls: burn::nn::Linear<B>,
    aux_reg: burn::nn::Linear<B>,
}

/// Configuration for the [`UNet`] model.
///
/// Sensible defaults are provided so you can construct the config with
/// `UNetConfig::new()` and tweak only what you need.
///
/// **`#[config(default = 32)]`:** Burn's Config derive lets you annotate
/// fields with defaults.  `UNetConfig::new()` uses these defaults; there's
/// no need to pass arguments (unlike PyTorch where you'd do
/// `UNet(base_channels=32)`).
#[derive(Config, Debug)]
pub struct UNetConfig {
    /// Number of filters in the first encoder stage.
    #[config(default = 32)]
    pub base_channels: usize,
}

impl UNetConfig {
    /// Initialise a [`UNet`] on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> UNet<B> {
        let c1 = self.base_channels; // 32
        let c2 = c1 * 2; // 64
        let c3 = c2 * 2; // 128

        let enc1 = ConvBlockConfig {
            in_channels: 3,
            out_channels: c1,
        }
        .init(device);

        let pool1 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        let enc2 = ConvBlockConfig {
            in_channels: c1,
            out_channels: c2,
        }
        .init(device);

        let pool2 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        let bottleneck = ConvBlockConfig {
            in_channels: c2,
            out_channels: c3,
        }
        .init(device);

        // Decoder stage 2: upsample bottleneck, concat with skip₂ → c3+c2
        // channels.
        let up_conv2 = Conv2dConfig::new([c3 + c2, c2], [1, 1]).init(device);
        let dec2 = ConvBlockConfig {
            in_channels: c2,
            out_channels: c2,
        }
        .init(device);

        // Decoder stage 1: upsample dec2 output, concat with skip₁ → c2+c1
        // channels.
        let up_conv1 = Conv2dConfig::new([c2 + c1, c1], [1, 1]).init(device);
        let dec1 = ConvBlockConfig {
            in_channels: c1,
            out_channels: c1,
        }
        .init(device);

        let seg_head = Conv2dConfig::new([c1, 1], [1, 1]).init(device);

        // Auxiliary heads for classification & regression on bottleneck
        // features.
        let aux_pool = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let aux_cls = burn::nn::LinearConfig::new(c3, 1).init(device);
        let aux_reg = burn::nn::LinearConfig::new(c3, 1).init(device);

        UNet {
            enc1,
            pool1,
            enc2,
            pool2,
            bottleneck,
            up_conv2,
            dec2,
            up_conv1,
            dec1,
            seg_head,
            aux_pool,
            aux_cls,
            aux_reg,
        }
    }
}

/// Bilinear-style 2× upsample: repeat each spatial element 2× in both
/// height and width using the `repeat_dim` operation.
///
/// Burn (0.20) doesn't have a built-in `F.interpolate()` equivalent, so we
/// use a reshape → repeat trick (pixel replication / nearest-neighbour
/// upsampling).  The result is mathematically equivalent to
/// `F.interpolate(x, scale_factor=2, mode='nearest')` in PyTorch.
fn upsample_2x<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, c, h, w] = x.dims();
    // Repeat along H then W.
    x.reshape([b, c, h, 1, w, 1])
        .repeat_dim(3, 2)
        .repeat_dim(5, 2)
        .reshape([b, c, h * 2, w * 2])
}

/// Outputs from the [`UNet`] forward pass.
#[derive(Debug)]
pub struct UNetOutput<B: Backend> {
    /// Segmentation logits `[B, 1, H, W]`.
    pub seg_logits: Tensor<B, 4>,
    /// Binary classification logits `[B, 1]`.
    pub cls_logits: Tensor<B, 2>,
    /// Regression predictions `[B, 1]`.
    pub reg_preds: Tensor<B, 2>,
}

impl<B: Backend> UNet<B> {
    /// Full forward pass returning segmentation logits and auxiliary outputs.
    pub fn forward(&self, x: Tensor<B, 4>) -> UNetOutput<B> {
        // ── Encoder ──────────────────────────────────────────────────────────
        let skip1 = self.enc1.forward(x); // [B, c1, H, W]
        let x = self.pool1.forward(skip1.clone()); // [B, c1, H/2, W/2]
        let skip2 = self.enc2.forward(x); // [B, c2, H/2, W/2]
        let x = self.pool2.forward(skip2.clone()); // [B, c2, H/4, W/4]

        // ── Bottleneck ───────────────────────────────────────────────────────
        let x = self.bottleneck.forward(x); // [B, c3, H/4, W/4]

        // ── Auxiliary heads (cls + reg) from bottleneck features ─────────────
        let aux = self.aux_pool.forward(x.clone()); // [B, c3, 1, 1]
        let [b, c3, _, _] = aux.dims();
        let aux_flat = aux.reshape([b, c3]); // [B, c3]
        let cls_logits = self.aux_cls.forward(aux_flat.clone()); // [B, 1]
        let reg_preds = self.aux_reg.forward(aux_flat); // [B, 1]

        // ── Decoder ──────────────────────────────────────────────────────────
        let x = upsample_2x(x); // [B, c3, H/2, W/2]
        let x = Tensor::cat(vec![x, skip2], 1); // [B, c3+c2, ...]
        let x = self.up_conv2.forward(x);
        let x = self.dec2.forward(x);

        let x = upsample_2x(x); // [B, c2, H, W]
        let x = Tensor::cat(vec![x, skip1], 1); // [B, c2+c1, ...]
        let x = self.up_conv1.forward(x);
        let x = self.dec1.forward(x);

        let seg_logits = self.seg_head.forward(x); // [B, 1, H, W]

        UNetOutput {
            seg_logits,
            cls_logits,
            reg_preds,
        }
    }

    /// Compute the combined training loss:
    ///   `loss = BCE(seg) + BCE(cls) + MSE(reg)`
    ///
    /// Returns a `RegressionOutput` (even though it mixes losses) because
    /// Burn's `TrainStep` / `InferenceStep` require a prediction and target
    /// tensor of the same shape.  We pack the scalar loss into a 1-element
    /// tensor.
    ///
    /// **Why `RegressionOutput` and not a custom type?**  Burn's
    /// `SupervisedTraining` expects `TrainStep::Output` to implement
    /// `Adaptor<LossInput>`.  The built-in `RegressionOutput` already does,
    /// so we piggyback on it.  In a real project you might implement your
    /// own `Adaptor` for richer metric logging.
    pub fn forward_step(
        &self,
        batch: SegmentationBatch<B>,
    ) -> RegressionOutput<B> {
        let output = self.forward(batch.images);

        // ── Segmentation loss (binary cross-entropy with logits) ─────────
        let seg_loss =
            binary_cross_entropy_with_logits(output.seg_logits, batch.masks);

        // ── Classification loss ──────────────────────────────────────────
        let cls_batch_size = batch.binary_targets.dims()[0];
        let cls_targets =
            batch.binary_targets.float().reshape([cls_batch_size, 1]);
        let cls_loss =
            binary_cross_entropy_with_logits(output.cls_logits, cls_targets);

        // ── Regression loss (MSE) ────────────────────────────────────────
        let reg_batch_size = batch.regression_targets.dims()[0];
        let reg_targets = batch.regression_targets.reshape([reg_batch_size, 1]);
        let reg_loss = (output.reg_preds - reg_targets.clone())
            .powf_scalar(2.0)
            .mean();

        // ── Combined loss ────────────────────────────────────────────────
        let loss = seg_loss + cls_loss + reg_loss.clone();

        // Pack into RegressionOutput with dummy output / targets tensors.
        RegressionOutput {
            loss,
            output: reg_loss.clone().reshape([1, 1]),
            targets: reg_loss.reshape([1, 1]),
        }
    }
}

// ── Binary cross-entropy with logits (numerically stable) ────────────────────

/// Numerically stable binary cross-entropy with logits.
///
/// `BCE(logits, targets) = mean( max(logits,0) - logits*targets +
/// log(1+exp(-|logits|)) )`
///
/// This is the same formula used by PyTorch's
/// `F.binary_cross_entropy_with_logits`. We implement it manually because Burn
/// (0.20) doesn't ship this loss function out of the box.
///
/// **Const generic `D`:** the function is generic over the tensor
/// dimensionality, so it works for both 4D segmentation logits (`[B,1,H,W]`)
/// and 2D classification logits (`[B,1]`).
fn binary_cross_entropy_with_logits<B: Backend, const D: usize>(
    logits: Tensor<B, D>,
    targets: Tensor<B, D>,
) -> Tensor<B, 1> {
    let zeros = logits.zeros_like();
    let pos_part = logits.clone().max_pair(zeros); // max(logits, 0)
    let neg_abs = logits.clone().abs().neg(); // -|logits|
    let log_term = neg_abs.exp().log1p(); // log(1 + exp(-|logits|))
    let loss = pos_part - logits * targets + log_term;
    loss.mean().reshape([1])
}

// ─────────────────────────────────────────────────────────────────────────────
// TrainStep / InferenceStep implementations
// ─────────────────────────────────────────────────────────────────────────────
//
// These traits are Burn's equivalent of PyTorch Lightning's
// `training_step()` and `validation_step()`.  The `Learner` training loop
// calls `TrainStep::step()` for every training batch and
// `InferenceStep::step()` for every validation batch.
//
// Notice the different trait bounds:
//   - `TrainStep` requires `B: AutodiffBackend` (so `.backward()` works)
//   - `InferenceStep` only requires `B: Backend` (no gradient tracking)
//
// This means you literally *cannot* accidentally compute gradients during
// validation — the type system prevents it.

impl<B: AutodiffBackend> TrainStep for UNet<B> {
    type Input = SegmentationBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(
        &self,
        item: SegmentationBatch<B>,
    ) -> TrainOutput<RegressionOutput<B>> {
        let result = self.forward_step(item);
        TrainOutput::new(self, result.loss.backward(), result)
    }
}

impl<B: Backend> burn::train::InferenceStep for UNet<B> {
    type Input = SegmentationBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, item: SegmentationBatch<B>) -> RegressionOutput<B> {
        self.forward_step(item)
    }
}
