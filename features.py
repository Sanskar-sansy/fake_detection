"""
features.py
============
Shared handcrafted feature-extraction functions used by both train.py and
predict.py. Keeping this logic in one file guarantees that training and
inference always compute features the exact same way (a very common bug
source is letting these two paths drift apart).

All functions operate on a BGR image as loaded by cv2.imread().
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Dict


# --------------------------------------------------------------------------
# 1. FREQUENCY-DOMAIN FEATURES
# --------------------------------------------------------------------------
# Why this matters: when you photograph a screen, you are photographing a
# grid of physical sub-pixels (an LCD/OLED panel) or a halftone-printed
# photo. That regular grid interacts with the camera's own sensor grid and
# produces moire interference patterns. Moire shows up as energy concen-
# trated at very specific, non-random frequencies in the 2D Fourier
# transform, instead of energy that fades smoothly from low to high
# frequency the way it does in a normal photo of a real, irregular scene.

def extract_fft_features(gray: np.ndarray) -> Dict[str, float]:
    """Compute FFT-magnitude-spectrum based features.

    Returns:
        dict with: fft_high_freq_ratio, fft_peak_to_avg, fft_radial_std,
        fft_radial_slope
    """
    # Resize to a fixed size so FFT bins are comparable across images
    img = cv2.resize(gray, (256, 256)).astype(np.float32)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    magnitude_log = np.log1p(magnitude)  # log scale, avoids huge dynamic range

    h, w = magnitude_log.shape
    cy, cx = h // 2, w // 2

    # Build radial distance map (distance of each freq bin from DC/center)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int32)
    r_max = r.max()

    # Radial energy profile: average magnitude at each radius
    radial_profile = np.zeros(r_max + 1, dtype=np.float64)
    radial_counts = np.zeros(r_max + 1, dtype=np.float64)
    np.add.at(radial_profile, r.ravel(), magnitude_log.ravel())
    np.add.at(radial_counts, r.ravel(), 1)
    radial_counts[radial_counts == 0] = 1
    radial_profile = radial_profile / radial_counts

    # High frequency ratio: energy in outer 50% radius vs total energy.
    # Natural photos: energy decays smoothly -> low ratio.
    # Screen recaptures: pixel-grid/moire injects extra high-freq energy.
    mid = r_max // 2
    high_energy = radial_profile[mid:].sum()
    total_energy = radial_profile.sum() + 1e-8
    fft_high_freq_ratio = float(high_energy / total_energy)

    # Peak-to-average: moire produces sharp narrow spikes in the spectrum.
    # A high max/mean ratio (excluding the DC component) signals such spikes.
    no_dc = radial_profile[2:]  # skip DC + immediate neighbour
    fft_peak_to_avg = float(no_dc.max() / (no_dc.mean() + 1e-8)) if len(no_dc) else 0.0

    # Radial std: how "spiky"/irregular the radial profile is.
    fft_radial_std = float(np.std(radial_profile))

    # Radial slope: fit a line to log-energy vs radius. Natural images have
    # a fairly consistent negative slope (1/f-ish falloff). Screen images
    # often deviate from this clean falloff.
    radii = np.arange(len(radial_profile))
    if len(radii) > 2:
        slope = float(np.polyfit(radii[1:], radial_profile[1:], 1)[0])
    else:
        slope = 0.0

    return {
        "fft_high_freq_ratio": fft_high_freq_ratio,
        "fft_peak_to_avg": fft_peak_to_avg,
        "fft_radial_std": fft_radial_std,
        "fft_radial_slope": slope,
    }


# --------------------------------------------------------------------------
# 2. TEXTURE FEATURES
# --------------------------------------------------------------------------
# Why this matters: re-photographing a flat screen produces images that are
# often slightly softer/blurrier (focus + an extra layer of glass/coating)
# or, conversely, have unnaturally crisp repeating edges (the pixel grid
# itself). Texture statistics like Laplacian variance and edge density pick
# up on this difference in "naturalness" of detail.

def extract_texture_features(gray: np.ndarray) -> Dict[str, float]:
    """Compute texture / sharpness statistics.

    Returns:
        dict with: laplacian_var, edge_density, grad_mean, grad_std
    """
    img = cv2.resize(gray, (512, 512))

    # Laplacian variance: a classic, fast blur/sharpness detector.
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian_var = float(laplacian.var())

    # Canny edge density: fraction of pixels that are edges.
    edges = cv2.Canny(img, 100, 200)
    edge_density = float(np.mean(edges > 0))

    # Gradient magnitude stats via Sobel.
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_mean = float(grad_mag.mean())
    grad_std = float(grad_mag.std())

    return {
        "laplacian_var": laplacian_var,
        "edge_density": edge_density,
        "grad_mean": grad_mean,
        "grad_std": grad_std,
    }


# --------------------------------------------------------------------------
# 3. COLOR FEATURES
# --------------------------------------------------------------------------
# Why this matters: screens (LCD/OLED) emit light through RGB sub-pixel
# filters and a backlight, which shifts color reproduction (commonly a
# slight blue/cyan cast, crushed blacks, or boosted saturation) compared to
# how a camera sensor captures real-world reflected light. HSV saturation
# stats are especially good at catching "too vivid"/over-saturated screen
# content or oddly flat printed photos.

def extract_color_features(bgr: np.ndarray) -> Dict[str, float]:
    """Compute RGB and HSV channel statistics.

    Returns:
        dict with per-channel mean/std for B,G,R and H,S,V plus
        saturation mean/std again called out explicitly.
    """
    img = cv2.resize(bgr, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    feats: Dict[str, float] = {}
    channel_names_bgr = ["b", "g", "r"]
    for i, name in enumerate(channel_names_bgr):
        ch = img[:, :, i].astype(np.float32)
        feats[f"{name}_mean"] = float(ch.mean())
        feats[f"{name}_std"] = float(ch.std())

    channel_names_hsv = ["h", "s", "v"]
    for i, name in enumerate(channel_names_hsv):
        ch = hsv[:, :, i].astype(np.float32)
        feats[f"{name}_mean"] = float(ch.mean())
        feats[f"{name}_std"] = float(ch.std())

    # Explicit saturation mean/std (duplicated naming for clarity per spec)
    feats["saturation_mean"] = feats["s_mean"]
    feats["saturation_std"] = feats["s_std"]

    return feats


# --------------------------------------------------------------------------
# 4. SCREEN-ARTIFACT-SPECIFIC FEATURES
# --------------------------------------------------------------------------
# Why this matters: beyond generic frequency/texture/color stats, screens
# leave a few very specific fingerprints:
#  - A periodic sub-pixel grid (periodicity score via autocorrelation)
#  - Localized repeating micro-texture (block-to-block self-similarity)
#  - Narrow, tall spikes in the FFT magnitude spectrum away from the
#    origin (true moire peaks, distinct from the broader radial stats
#    above, since moire is often NOT radially symmetric)

def extract_screen_artifact_features(gray: np.ndarray) -> Dict[str, float]:
    """Compute features specifically targeting screen-recapture artifacts.

    Returns:
        dict with: periodicity_score, local_repetition_score,
        moire_peak_count, moire_peak_strength
    """
    img = cv2.resize(gray, (256, 256)).astype(np.float32)

    # --- Periodicity score via autocorrelation -----------------------
    # Compute normalized autocorrelation using FFT (Wiener-Khinchin).
    f = np.fft.fft2(img - img.mean())
    power = np.abs(f) ** 2
    autocorr = np.fft.ifft2(power).real
    autocorr = np.fft.fftshift(autocorr)
    autocorr /= (autocorr.max() + 1e-8)

    h, w = autocorr.shape
    cy, cx = h // 2, w // 2
    # Look at a ring of radius 4-30 px around the center (skip the
    # trivial central peak at lag 0). Strong secondary peaks here imply
    # a repeating sub-pixel/grid pattern.
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    ring_mask = (r >= 4) & (r <= 30)
    periodicity_score = float(autocorr[ring_mask].max()) if ring_mask.any() else 0.0

    # --- Local repetition score ---------------------------------------
    # Split image into non-overlapping blocks, compare each block to its
    # right/bottom neighbour via normalized correlation. Real-world
    # textures rarely repeat block-to-block; pixel grids/halftone
    # patterns do.
    block = 16
    n_blocks_y, n_blocks_x = h // block, w // block
    sims = []
    for by in range(n_blocks_y - 1):
        for bx in range(n_blocks_x - 1):
            b1 = img[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
            b2 = img[by * block:(by + 1) * block, (bx + 1) * block:(bx + 2) * block]
            b1f, b2f = b1.flatten(), b2.flatten()
            if b1f.std() < 1e-3 or b2f.std() < 1e-3:
                continue
            corr = np.corrcoef(b1f, b2f)[0, 1]
            if not np.isnan(corr):
                sims.append(corr)
    local_repetition_score = float(np.mean(sims)) if sims else 0.0

    # --- Moire peak detection in FFT magnitude spectrum ----------------
    fshift = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log1p(np.abs(fshift))
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

    # Mask out the central DC blob (low frequencies always dominate).
    mask = np.ones_like(mag_norm)
    cv2.circle(mask, (cx, cy), 12, 0, -1)
    masked = mag_norm * mask

    threshold = masked.mean() + 4 * masked.std()
    peak_mask = masked > threshold
    moire_peak_count = int(peak_mask.sum())
    moire_peak_strength = float(masked[peak_mask].sum()) if moire_peak_count > 0 else 0.0

    return {
        "periodicity_score": periodicity_score,
        "local_repetition_score": local_repetition_score,
        "moire_peak_count": float(moire_peak_count),
        "moire_peak_strength": moire_peak_strength,
    }


# --------------------------------------------------------------------------
# 5. MASTER FEATURE EXTRACTOR
# --------------------------------------------------------------------------

def extract_features(image_path: str) -> Dict[str, float]:
    """Load an image and compute the full handcrafted feature vector.

    Args:
        image_path: path to an image file (jpg/png/etc.)

    Returns:
        dict mapping feature name -> float value. The ORDER of keys is
        stable (Python 3.7+ dicts preserve insertion order) and is used
        directly to build the feature matrix in train.py / predict.py.

    Raises:
        ValueError: if the image cannot be read by OpenCV.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    feats: Dict[str, float] = {}
    feats.update(extract_fft_features(gray))
    feats.update(extract_texture_features(gray))
    feats.update(extract_color_features(bgr))
    feats.update(extract_screen_artifact_features(gray))
    return feats


def feature_names() -> list:
    """Return the canonical ordered list of feature names.

    Computed once on a tiny dummy image so train.py / predict.py can both
    rely on the exact same ordering without re-deriving it.
    """
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    gray = cv2.cvtColor(dummy, cv2.COLOR_BGR2GRAY)
    feats = {}
    feats.update(extract_fft_features(gray))
    feats.update(extract_texture_features(gray))
    feats.update(extract_color_features(dummy))
    feats.update(extract_screen_artifact_features(gray))
    return list(feats.keys())
