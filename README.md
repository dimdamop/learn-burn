# learn-burn

An educational project demonstrating how to train and run inference with a
**UNet-like segmentation model** using the [Burn](https://burn.dev) deep-learning
framework in Rust.

**Target audience:** experienced Python / PyTorch developers who want to learn
Rust + Burn by example.

---

## What This Project Covers

| Concept | Where |
|---|---|
| Defining a custom **Dataset** (Burn's `Dataset` trait ≈ PyTorch's `Dataset`) | `dataset.rs` |
| Data transforms via `Mapper` (≈ `torchvision.transforms`) composed with `MapperDataset` | `data.rs` |
| Collating samples into batches (`Batcher` trait ≈ PyTorch's `collate_fn`) | `data.rs` |
| Building a CNN architecture with `#[derive(Module)]` (≈ `nn.Module`) | `model.rs` |
| Config-based initialization with `#[derive(Config)]` (like a typed dict you can serialize) | `model.rs`, `training.rs` |
| Training loop via Burn's `Learner` + `SupervisedTraining` (≈ Lightning / Ignite) | `training.rs` |
| Model serialization and loading (`Recorder` ≈ `torch.save` / `torch.load`) | `training.rs`, `inference.rs` |
| Swapping backends (CPU / GPU / WGPU) via Cargo features, zero code changes | `main.rs` |
| Automatic differentiation with `Autodiff<B>` (≈ `torch.autograd`) | `main.rs`, `training.rs` |

---

## Project Structure

```
learn-burn/
├── Cargo.toml           # Dependencies & feature flags
├── setup-conda.sh       # Self-contained environment setup (conda + libtorch)
├── conda-init.yaml      # Conda env spec
├── README.md            # You are here
└── src/
    ├── lib.rs           # Crate root – module declarations
    ├── dataset.rs       # Synthetic ellipse-segmentation dataset
    ├── data.rs          # Transforms (Normalize, HorizontalFlip) & Batcher
    ├── model.rs         # UNet architecture + loss + TrainStep/InferenceStep
    ├── training.rs      # Training pipeline (config + Learner)
    ├── inference.rs     # Load model, run predictions
    └── main.rs          # CLI entry point, backend dispatch
```

---

## Architecture

```
 Input [B, 3, 96, 96]
   │
   ▼
 ┌──────────────┐
 │ Encoder Blk 1│──► skip₁ [B, 32, 96, 96]
 └──────┬───────┘
        │ MaxPool 2×2
 ┌──────▼───────┐
 │ Encoder Blk 2│──► skip₂ [B, 64, 48, 48]
 └──────┬───────┘
        │ MaxPool 2×2
 ┌──────▼───────┐
 │  Bottleneck  │       [B, 128, 24, 24]
 └──────┬───────┘
        │               ┌─ AdaptiveAvgPool → Linear → cls logit
        │               └─ AdaptiveAvgPool → Linear → reg value
        │ Upsample 2×
 ┌──────▼───────┐
 │ Decoder Blk 2│◄── cat(skip₂)
 └──────┬───────┘
        │ Upsample 2×
 ┌──────▼───────┐
 │ Decoder Blk 1│◄── cat(skip₁)
 └──────┬───────┘
        │
 ┌──────▼──────┐
 │  1×1 Conv   │ → [B, 1, 96, 96]  seg logits
 └─────────────┘
```

Each encoder/decoder block: `Conv2d(3×3) → BatchNorm → ReLU → Conv2d(3×3) → BatchNorm → ReLU`.

---

## Quick Start

### 1. Environment Setup (libtorch via conda + pip)

```bash
# Creates a self-contained conda env under ./conda_env with PyTorch + CUDA.
bash setup-conda.sh

# Activate the environment and set libtorch paths:
source setup-conda.sh        # sets LIBTORCH, LD_LIBRARY_PATH, LIBTORCH_CXX11_ABI
```

**What the env vars do:**

| Variable | Purpose |
|---|---|
| `LIBTORCH` | Tells the `tch` (torch-sys) crate where to find libtorch headers & libraries |
| `LD_LIBRARY_PATH` | So the dynamic linker finds `libtorch_cpu.so`, `libc10_cuda.so`, etc. at runtime |
| `LIBTORCH_CXX11_ABI` | Must be `1` when using pip-installed PyTorch (which uses the C++11 ABI). If wrong, you'll get cryptic linker errors about missing `std::__cxx11::basic_string` symbols |

### 2. Train

```bash
# GPU (default — tch-gpu feature)
cargo run --release -- train --artifact-dir artifacts

# CPU only (tch backend, no CUDA)
cargo run --release --no-default-features --features tch-cpu -- train

# Pure-Rust CPU (no libtorch needed)
cargo run --release --no-default-features --features ndarray -- train
```

> **Important:** always use `--release`. Debug builds are 50–100× slower for
> numerical workloads — a 30-second training run becomes a multi-hour ordeal.

### 3. Inference

```bash
cargo run --release -- infer --artifact-dir artifacts --num-samples 10
```

---

## Key Burn Concepts for Python Developers

### Backend Generics (no equivalent in PyTorch)

In Burn, every tensor, module, and training function is generic over a
**Backend** `B`:

```rust
fn train<B: AutodiffBackend>(device: B::Device) { ... }
```

This is like writing code that works on `torch.Tensor` without hard-coding
whether it lives on CPU or CUDA — except the dispatch happens at **compile
time** via monomorphization, not at runtime. The concrete backend is selected
once in `main.rs` via Cargo features.

### `#[derive(Module)]` ≈ `nn.Module`

```rust
#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv1: Conv2d<B>,   // ← sub-modules are struct fields
    bn1: BatchNorm<B>,
    // ...
}
```

**What the derive macro gives you for free:**
- Parameter collection (like `model.parameters()`)
- Device transfer (like `.to(device)`)
- Serialization / deserialization (`model.save_file(...)` / `load_record(...)`)

**What you still write by hand:** the `forward()` method — Burn doesn't have
a magic `__call__` / `forward()` trait.

### `#[derive(Config)]` ≈ Typed `dict` / `dataclass`

```rust
#[derive(Config, Debug)]
pub struct UNetConfig {
    #[config(default = 32)]
    pub base_channels: usize,
}
```

A Config is a plain data struct that can be serialized to JSON, loaded back,
and used to initialize the corresponding Module. It separates the
*hyperparameters* (Config) from the *learned weights* (Module).

Convention: each module `Foo<B>` has a companion `FooConfig` with an
`init(&self, device) -> Foo<B>` method.

### `Dataset` / `Batcher` / `MapperDataset` ≈ PyTorch DataLoader stack

| Burn | PyTorch | Purpose |
|---|---|---|
| `Dataset<Item>` trait | `torch.utils.data.Dataset` | Index into data, return one item |
| `Mapper<In, Out>` trait | `torchvision.transforms` | Transform a single item |
| `MapperDataset` | `TransformDataset` | Compose transforms onto a dataset |
| `Batcher<B, Item, Batch>` trait | `collate_fn` | Combine items into a batch of tensors |
| `DataLoaderBuilder` | `DataLoader(...)` | Shuffling, batching, multi-worker loading |

### `Autodiff<B>` ≈ `torch.autograd`

`Autodiff<B>` is a wrapper backend that records operations for reverse-mode
AD. When you see `type TrainBackend = Autodiff<LibTorch>`, it means "LibTorch
as the compute engine, with Burn's autodiff tape on top."

Calling `loss.backward()` returns gradients, exactly like PyTorch, but the
type system enforces that only `AutodiffBackend` tensors can be
differentiated — you can't accidentally call `.backward()` on an inference
tensor.

### `TrainStep` / `InferenceStep` traits

These are the contract that Burn's `Learner` expects:

```rust
impl<B: AutodiffBackend> TrainStep for UNet<B> {
    type Input = SegmentationBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: SegmentationBatch<B>) -> TrainOutput<...> {
        let result = self.forward_step(batch);
        TrainOutput::new(self, result.loss.backward(), result)
    }
}
```

Think of it like `training_step()` in PyTorch Lightning — you receive a
batch, compute the loss, call `.backward()`, and return everything the
Learner needs to log metrics and update weights.

---

## Data Pipeline Flow

```
SyntheticEllipseDataset
  │  .get(i) → EllipseItem { image: Vec<f32>, mask: Vec<f32>, ... }
  │
  ├─ MapperDataset<NormalizeTransform>     [0,255] → [0,1]
  ├─ MapperDataset<FlipHorizontalTransform> 50% mirror
  │
  ▼
DataLoaderBuilder
  │  shuffles, batches, multi-worker
  │
  ▼
SegmentationBatcher.batch(Vec<EllipseItem>) → SegmentationBatch<B>
  │  flat buffers → Tensor on device
  │
  ▼
model.forward_step(batch) → loss
```

**Why `Vec<f32>` instead of tensors in dataset items?**

Burn datasets must produce `Clone + Send` items (for multi-worker loading).
Tensors are device-bound and not cheaply cloneable across threads. So the
convention is: keep data as plain Rust types in the dataset, and move to
tensors only inside the `Batcher`.

---

## Common Pitfalls

| Problem | Cause | Fix |
|---|---|---|
| Linker errors about `std::__cxx11::basic_string` | ABI mismatch between pip PyTorch (CXX11 ABI) and `tch` build | Set `LIBTORCH_CXX11_ABI=1` and do `cargo clean` |
| Training takes hours | Running in debug mode | Always use `cargo run --release` |
| `gen` is not an identifier | Rust nightly ≥ 1.96 reserves `gen` as a keyword | Use `gen_range()` instead of `gen::<T>()` |
| Shapes don't match after upsample+cat | Spatial dimensions not evenly divisible by pool stride | Use image sizes divisible by 4 (both pool layers) |

---

## License

Educational project — use freely.
