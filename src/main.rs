//! # learn-burn CLI
//!
//! Entry point that dispatches to `train` or `infer` sub-commands.
//!
//! ## Backend selection via Cargo features
//!
//! Burn is backend-agnostic: every tensor and module is generic over a
//! `Backend` trait.  The **concrete backend is chosen here** via `#[cfg]`
//! conditional compilation blocks, driven by Cargo features.
//!
//! This means:
//! - **Zero runtime overhead** — no dynamic dispatch, no `if device == "cuda"`.
//! - **Compile-time safety** — you can't accidentally mix CPU and GPU tensors.
//! - **Swap backends without touching model code** — just change the feature
//!   flag.
//!
//! | Feature | Backend | Device | Notes |
//! |---------|---------|--------|-------|
//! | `tch-gpu` (default) | `LibTorch` | `Cuda(0)` | Needs libtorch + CUDA |
//! | `tch-cpu` | `LibTorch` | `Cpu` | Needs libtorch |
//! | `ndarray` | `NdArray` | `Cpu` | Pure Rust, no external deps |
//! | `wgpu` | `Wgpu` | default GPU | Vulkan/Metal/DX12, cross-platform |
//!
//! **`Autodiff<B>`:** For training, the backend is wrapped in `Autodiff<B>`
//! which records operations for reverse-mode automatic differentiation
//! (like `torch.autograd`).  Inference uses the bare backend since no
//! gradients are needed.
//!
//! ## Quick start
//!
//! ```bash
//! # Train (tch backend, GPU — default)
//! cargo run --release -- train --artifact-dir artifacts
//!
//! # Train (ndarray backend, CPU)
//! cargo run --release --no-default-features --features ndarray -- train
//!
//! # Inference
//! cargo run --release -- infer --artifact-dir artifacts --num-samples 10
//! ```

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "learn-burn", about = "UNet segmentation with Burn")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Train the UNet model on synthetic data.
    Train {
        /// Directory for checkpoints, config, and the final model.
        #[arg(long, default_value = "artifacts")]
        artifact_dir: String,
        /// Path to a JSON config file. If omitted, uses defaults.
        #[arg(long)]
        config: Option<String>,
    },
    /// Run inference with a previously trained model.
    Infer {
        /// Directory that contains the saved model.
        #[arg(long, default_value = "artifacts")]
        artifact_dir: String,
        /// Number of test samples to run.
        #[arg(long, default_value_t = 10)]
        num_samples: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    // ── tch backend (libtorch) — default, supports CUDA GPU ──────────────
    #[cfg(feature = "tch-gpu")]
    {
        type TrainBackend = burn::backend::Autodiff<burn::backend::LibTorch>;
        type InferBackend = burn::backend::LibTorch;

        println!(
            "CUDA available: {}, devices: {}",
            tch::Cuda::is_available(),
            tch::Cuda::device_count()
        );

        let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);

        match cli.command {
            Command::Train {
                artifact_dir,
                config,
            } => {
                learn_burn::training::run::<TrainBackend>(
                    &artifact_dir,
                    device,
                    config.as_deref(),
                );
            }
            Command::Infer {
                artifact_dir,
                num_samples,
            } => {
                learn_burn::inference::run::<InferBackend>(
                    &artifact_dir,
                    device,
                    num_samples,
                );
            }
        }
        return;
    }

    #[cfg(feature = "tch-cpu")]
    {
        type TrainBackend = burn::backend::Autodiff<burn::backend::LibTorch>;
        type InferBackend = burn::backend::LibTorch;

        let device = burn::backend::libtorch::LibTorchDevice::Cpu;

        match cli.command {
            Command::Train {
                artifact_dir,
                config,
            } => {
                learn_burn::training::run::<TrainBackend>(
                    &artifact_dir,
                    device,
                    config.as_deref(),
                );
            }
            Command::Infer {
                artifact_dir,
                num_samples,
            } => {
                learn_burn::inference::run::<InferBackend>(
                    &artifact_dir,
                    device,
                    num_samples,
                );
            }
        }
        return;
    }

    // ── ndarray backend (pure-Rust CPU) ──────────────────────────────────
    #[cfg(feature = "ndarray")]
    {
        type TrainBackend = burn::backend::Autodiff<burn::backend::NdArray>;
        type InferBackend = burn::backend::NdArray;

        match cli.command {
            Command::Train {
                artifact_dir,
                config,
            } => {
                learn_burn::training::run::<TrainBackend>(
                    &artifact_dir,
                    burn::backend::ndarray::NdArrayDevice::Cpu,
                    config.as_deref(),
                );
            }
            Command::Infer {
                artifact_dir,
                num_samples,
            } => {
                learn_burn::inference::run::<InferBackend>(
                    &artifact_dir,
                    burn::backend::ndarray::NdArrayDevice::Cpu,
                    num_samples,
                );
            }
        }
        return;
    }

    // ── wgpu backend (Vulkan/Metal/DX12 GPU) ────────────────────────────
    #[cfg(feature = "wgpu")]
    {
        type TrainBackend = burn::backend::Autodiff<burn::backend::Wgpu>;
        type InferBackend = burn::backend::Wgpu;

        match cli.command {
            Command::Train {
                artifact_dir,
                config,
            } => {
                learn_burn::training::run::<TrainBackend>(
                    &artifact_dir,
                    burn::backend::wgpu::WgpuDevice::default(),
                    config.as_deref(),
                );
            }
            Command::Infer {
                artifact_dir,
                num_samples,
            } => {
                learn_burn::inference::run::<InferBackend>(
                    &artifact_dir,
                    burn::backend::wgpu::WgpuDevice::default(),
                    num_samples,
                );
            }
        }
    }
}
