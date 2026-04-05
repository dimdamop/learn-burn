//! # Synthetic Ellipse-Segmentation Dataset
//!
//! This module generates a synthetic dataset that mirrors the Python reference
//! implementation.  Each sample consists of:
//!
//! * A random RGB image (height × width × 3) with several overlaid ellipses of
//!   varying intensity.
//! * A binary segmentation mask for a specific "ground-truth" ellipse layer.
//! * A scalar binary classification target (1 if the masked region's mean
//!   intensity is < 128, else 0).
//! * A scalar regression target (masked-region mean − whole-image mean).
//!
//! The [`SyntheticEllipseDataset`] struct implements Burn's [`Dataset`] trait
//! so it can be plugged directly into `DataLoaderBuilder`.
//!
//! ## Python ↔ Rust key differences
//!
//! **Python → Rust equivalences:**
//!
//! - `class MyDataset(Dataset): def __getitem__` → `impl Dataset<EllipseItem>
//!   for ... { fn get() }`
//! - Items can be dicts with tensors → Items must be `Clone + Send` plain
//!   structs (for multi-worker loading)
//! - `np.array` / `torch.Tensor` in the dataset → `Vec<f32>` — tensors created
//!   later in `Batcher`
//! - Thread safety is implicit (GIL) → `Mutex<Vec<Option<...>>>` for the cache
//!   because `get(&self)` is `&`-only

use burn::data::dataset::Dataset;
use image::imageops::FilterType;
use image::RgbImage;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Mutex;

// ── Constants that mirror the Python reference ───────────────────────────────

/// Minimum spatial dimension for the generated images.
pub const MIN_IMG_LEN: usize = 80;
/// Maximum spatial dimension for the generated images.
pub const MAX_IMG_LEN: usize = 120;
/// Number of ellipse layers drawn on every image.
pub const NUM_LAYERS: usize = 8;
/// Which ellipse layer provides the ground-truth segmentation mask (0-indexed).
pub const GT_LAYER_IDX: usize = 5;

/// Fixed image size used after resizing (height, width).
/// All images are resized to this size so that they can be batched into a
/// single tensor.
pub const FIXED_H: usize = 96;
pub const FIXED_W: usize = 96;

// ── Per-sample item ──────────────────────────────────────────────────────────

/// A single sample from the synthetic dataset.
///
/// All pixel buffers are stored as flat `Vec<f32>` in **row-major** order.
/// Shapes:
/// - `image`:  `[3, FIXED_H, FIXED_W]`  (CHW layout, values in 0..255)
/// - `mask`:   `[1, FIXED_H, FIXED_W]`  (0.0 or 1.0)
///
/// **Why `Vec<f32>` and not a tensor?**  Burn datasets must produce items
/// that are `Clone + Send` for multi-worker data loading.  Tensors are tied
/// to a specific backend and device, which makes them awkward to clone
/// across threads.  The conversion to tensors happens in the `Batcher`
/// (see `data.rs`).
///
/// **Why `#[derive(Serialize, Deserialize)]`?**  Burn's `SqliteDataset` and
/// caching infrastructure require items to be serializable.  Even though we
/// don't use SQLite here, it's good practice to derive these traits.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EllipseItem {
    /// RGB image flattened in CHW order, values in [0, 255].
    pub image: Vec<f32>,
    /// Binary mask flattened in CHW order (single channel), values 0.0 / 1.0.
    pub mask: Vec<f32>,
    /// Binary classification target: 1 if masked-region mean < 128, else 0.
    pub binary_target: i32,
    /// Regression target: masked-region mean − whole-image mean.
    pub regression_target: f32,
    /// Original height before resize (for reference).
    pub original_height: usize,
    /// Original width before resize (for reference).
    pub original_width: usize,
}

// ── Ellipse drawing helper ───────────────────────────────────────────────────

/// Draws a filled ellipse on `img` (HWC layout, i32) and returns the boolean
/// mask.  Mirrors the Python `ellipse()` function.
///
/// **Rust ownership note:** `img` is passed as `&mut [i32]` (a mutable
/// slice), which is Rust's equivalent of modifying a NumPy array in-place.
/// The caller retains ownership of the backing `Vec`; we just borrow it
/// mutably for the duration of this call.
fn draw_ellipse(
    img: &mut [i32],
    height: usize,
    width: usize,
    channels: usize,
    hlen_frac: f64,
    vlen_frac: f64,
    intensity: i32,
    rng: &mut StdRng,
) -> Vec<bool> {
    let img_size = height.min(width) as f64;
    let hlen = (img_size * hlen_frac) as usize;
    let vlen = (img_size * vlen_frac) as usize;
    let hrad = (hlen / 2).max(1).min(width / 2);
    let vrad = (vlen / 2).max(1).min(height / 2);

    // Centre is sampled so the ellipse stays fully within the image.
    let ctr_h = rng.gen_range(hrad..(width - hrad).max(hrad + 1));
    let ctr_v = rng.gen_range(vrad..(height - vrad).max(vrad + 1));

    let mut mask = vec![false; height * width];

    for y in 0..height {
        for x in 0..width {
            let dx = (x as f64 - ctr_h as f64) / hrad as f64;
            let dy = (y as f64 - ctr_v as f64) / vrad as f64;
            if dx * dx + dy * dy <= 1.0 {
                mask[y * width + x] = true;
                let base = (y * width + x) * channels;
                for c in 0..channels {
                    img[base + c] = intensity;
                }
            }
        }
    }

    mask
}

// ── Resize helpers (using the `image` crate) ─────────────────────────────────

/// Resize an RGB image (HWC u8) to the target size using **Lanczos3**
/// filtering, then return a CHW f32 buffer.
///
/// We use the [`image`] crate (Rust's equivalent of Pillow) for high-quality
/// resizing.  The workflow is:  `i32 HWC buffer → image::RgbImage → resize()
/// → CHW f32`.  This is like doing `Image.fromarray(arr).resize(size,
/// Image.LANCZOS)` in Python.
fn resize_rgb_to_chw(
    src: &[i32],
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
) -> Vec<f32> {
    // Build an RgbImage (HWC, u8) from the i32 HWC buffer.
    let mut rgb = RgbImage::new(w_in as u32, h_in as u32);
    for y in 0..h_in {
        for x in 0..w_in {
            let base = (y * w_in + x) * 3;
            let r = src[base].clamp(0, 255) as u8;
            let g = src[base + 1].clamp(0, 255) as u8;
            let b = src[base + 2].clamp(0, 255) as u8;
            rgb.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    let resized = image::imageops::resize(
        &rgb,
        w_out as u32,
        h_out as u32,
        FilterType::Lanczos3,
    );

    // Convert resized HWC u8 → CHW f32.
    let mut chw = vec![0.0f32; 3 * h_out * w_out];
    for y in 0..h_out {
        for x in 0..w_out {
            let px = resized.get_pixel(x as u32, y as u32).0;
            for c in 0..3 {
                chw[c * h_out * w_out + y * w_out + x] = px[c] as f32;
            }
        }
    }
    chw
}

/// Resize a single-channel binary mask to the target size using **nearest-
/// neighbour** interpolation (to preserve sharp binary edges), then return a
/// CHW f32 buffer.
///
/// Nearest-neighbour is essential for masks — Lanczos/bilinear would blur the
/// {0, 1} boundary into intermediate values, ruining the ground truth.
fn resize_mask_to_chw(
    mask: &[bool],
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
) -> Vec<f32> {
    let buf: Vec<f32> = mask
        .iter()
        .map(|&m| if m { 1.0f32 } else { 0.0f32 })
        .collect();
    let img: image::ImageBuffer<image::Luma<f32>, Vec<f32>> =
        image::ImageBuffer::from_raw(
            w_in as u32,
            h_in as u32,
            buf,
        )
        .expect("mask buffer size mismatch");

    let resized = image::imageops::resize(
        &img,
        w_out as u32,
        h_out as u32,
        FilterType::Nearest,
    );

    resized.into_raw()
}

// ── Sample generation ────────────────────────────────────────────────────────

/// Generate one [`EllipseItem`] using the given random number generator.
/// This is the Rust equivalent of the Python `sample_element()` function.
fn sample_element(rng: &mut StdRng) -> EllipseItem {
    let height = rng.gen_range(MIN_IMG_LEN..MAX_IMG_LEN);
    let width = rng.gen_range(MIN_IMG_LEN..MAX_IMG_LEN);
    let channels = 3usize;
    let num_pixels = height * width * channels;

    // Random background image (HWC, i32).
    let mut img: Vec<i32> =
        (0..num_pixels).map(|_| rng.gen_range(0..255)).collect();

    // Sorted random intensities for each ellipse layer.
    let mut layer_vals: Vec<i32> =
        (0..NUM_LAYERS).map(|_| rng.gen_range(0..255)).collect();
    layer_vals.sort();

    let mut gt_mask = vec![false; height * width];

    for (layer_idx, &layer_val) in layer_vals.iter().enumerate() {
        let hlen_frac = rng.gen_range(0.0..1.0_f64) / 2.0 + 0.1; // 0.1 .. 0.6
        let vlen_frac = hlen_frac; // same as Python default (vlen = hlen)
        let mask = draw_ellipse(
            &mut img, height, width, channels, hlen_frac, vlen_frac, layer_val,
            rng,
        );
        if layer_idx == GT_LAYER_IDX {
            gt_mask = mask;
        }
    }

    // Compute masked-region mean and image mean (over all channels).
    let mut masked_sum: f64 = 0.0;
    let mut masked_count: usize = 0;
    for (i, &m) in gt_mask.iter().enumerate() {
        if m {
            let base = i * channels;
            for c in 0..channels {
                masked_sum += img[base + c] as f64;
                masked_count += 1;
            }
        }
    }
    let masked_mean = if masked_count > 0 {
        masked_sum / masked_count as f64
    } else {
        0.0
    };

    let img_mean: f64 =
        img.iter().map(|v| *v as f64).sum::<f64>() / num_pixels as f64;

    let binary_target = if masked_mean < 128.0 { 1 } else { 0 };
    let regression_target = (masked_mean - img_mean) as f32;

    // Resize image (HWC i32 → CHW f32) using Lanczos3 filtering via the `image`
    // crate.
    let resized_img = resize_rgb_to_chw(&img, height, width, FIXED_H, FIXED_W);

    // Resize mask using nearest-neighbour to keep sharp binary edges.
    let resized_mask =
        resize_mask_to_chw(&gt_mask, height, width, FIXED_H, FIXED_W);

    EllipseItem {
        image: resized_img,
        mask: resized_mask,
        binary_target,
        regression_target,
        original_height: height,
        original_width: width,
    }
}

// ── Dataset struct ───────────────────────────────────────────────────────────

/// A lazily-generated synthetic dataset.
///
/// Items are generated on the fly in [`Dataset::get`] using a **seeded RNG**
/// so that results are reproducible.  The dataset length is fixed at
/// construction time.
///
/// ## How it works
///
/// Each index is mapped to a deterministic seed (`base_seed + index`), so
/// `get(42)` always returns the same sample regardless of access order.
/// Generated items are cached behind a `Mutex` so repeated accesses are
/// cheap.  The `Mutex` is needed because the `Dataset` trait's `get()`
/// takes `&self` (shared reference), not `&mut self`, yet we need interior
/// mutability to populate the cache.  This is a common Rust pattern called
/// **interior mutability** — in Python you'd just mutate `self._cache`
/// freely.
///
/// ## Why not `RefCell`?
///
/// `RefCell` is single-threaded.  Burn's `DataLoaderBuilder` can spawn
/// multiple worker threads, each calling `get()` concurrently, so we need
/// `Mutex` (the thread-safe version).
///
/// # Example
///
/// ```rust
/// use learn_burn::dataset::SyntheticEllipseDataset;
/// use burn::data::dataset::Dataset;
///
/// let ds = SyntheticEllipseDataset::new(1000, 42);
/// assert_eq!(ds.len(), 1000);
/// let item = ds.get(0).unwrap();
/// ```
pub struct SyntheticEllipseDataset {
    /// Number of samples in the dataset.
    size: usize,
    /// Base seed – each index derives its own seed deterministically.
    base_seed: u64,
    /// Cache: we store generated items so repeated calls to `get(i)` are fast.
    cache: Mutex<Vec<Option<EllipseItem>>>,
}

impl SyntheticEllipseDataset {
    /// Creates a new synthetic dataset with `size` samples and the given base
    /// random seed.
    pub fn new(size: usize, seed: u64) -> Self {
        Self {
            size,
            base_seed: seed,
            cache: Mutex::new(vec![None; size]),
        }
    }

    /// Convenience constructors that mirror the conventional train / validation
    /// / test split.  They use different seeds so the splits are disjoint.
    pub fn train(size: usize) -> Self {
        Self::new(size, 0)
    }
    pub fn validation(size: usize) -> Self {
        Self::new(size, 1_000_000)
    }
    pub fn test(size: usize) -> Self {
        Self::new(size, 2_000_000)
    }
}

impl Dataset<EllipseItem> for SyntheticEllipseDataset {
    fn get(&self, index: usize) -> Option<EllipseItem> {
        if index >= self.size {
            return None;
        }

        // Fast path: return cached item.
        {
            let cache = self.cache.lock().unwrap();
            if let Some(item) = &cache[index] {
                return Some(item.clone());
            }
        }

        // Slow path: generate and cache.
        let seed = self.base_seed.wrapping_add(index as u64);
        let mut rng = StdRng::seed_from_u64(seed);
        let item = sample_element(&mut rng);

        let mut cache = self.cache.lock().unwrap();
        cache[index] = Some(item.clone());
        Some(item)
    }

    fn len(&self) -> usize {
        self.size
    }
}
