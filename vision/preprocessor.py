"""
Image Preprocessing and Enhancement Module for Indian ANPR.

Provides:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Bilateral filtering (edge-preserving denoising)
- Laplacian variance blur estimation and sharpness scoring
- Gamma correction (auto and manual) for low-light/night conditions
- Adaptive binarization and thresholding for license plate character isolation
"""

from typing import Tuple, Optional
import numpy as np
import cv2


def clahe_enhance(
    image: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE contrast enhancement.
    If image is 3-channel BGR, enhancement is applied to the L (Luminance) channel in LAB space.
    If image is 1-channel grayscale, enhancement is applied directly.
    """
    if image is None or image.size == 0:
        return image

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        gray = image.reshape(image.shape[0], image.shape[1])
        return clahe.apply(gray)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_enhanced = clahe.apply(l_chan)
        lab_enhanced = cv2.merge([l_enhanced, a_chan, b_chan])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return image


def bilateral_denoise(
    image: np.ndarray,
    d: int = 7,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
) -> np.ndarray:
    """
    Apply edge-preserving bilateral filtering to reduce sensor noise without blurring text edges.
    """
    if image is None or image.size == 0:
        return image
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def laplacian_variance(image: np.ndarray) -> float:
    """
    Calculate Laplacian variance sharpness metric: Var(Laplacian(I)).
    Higher values indicate sharp, in-focus images; low values indicate motion or defocus blur.
    """
    if image is None or image.size == 0:
        return 0.0

    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        gray = image
    else:
        gray = image.squeeze()

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(lap.var())
    return variance


def is_blurry(image: np.ndarray, threshold: float = 100.0) -> Tuple[bool, float]:
    """
    Determine whether an image is blurry based on Laplacian variance threshold.

    Returns:
        Tuple[bool, float]: (is_blurry, variance_value)
    """
    var = laplacian_variance(image)
    return (var < threshold, var)


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction non-linear luminance adjustment.
    gamma < 1.0 brightens shadows (useful for night captures).
    gamma > 1.0 darkens highlights (useful for sun glare).
    """
    if image is None or image.size == 0 or abs(gamma - 1.0) < 1e-4:
        return image

    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def auto_gamma(image: np.ndarray, target_mean: float = 128.0) -> np.ndarray:
    """
    Automatically adjust gamma to reach target mean brightness.
    """
    if image is None or image.size == 0:
        return image

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    current_mean = float(np.mean(gray))
    if current_mean <= 1.0:
        return gamma_correction(image, gamma=0.3)
    if current_mean >= 254.0:
        return gamma_correction(image, gamma=2.5)

    # (current_mean / 255.0) ** gamma = target_mean / 255.0
    gamma = np.log(target_mean / 255.0) / np.log(current_mean / 255.0)
    gamma = float(np.clip(gamma, 0.25, 3.0))
    return gamma_correction(image, gamma=gamma)


def binarize_plate(
    image: np.ndarray,
    method: str = "otsu",
    block_size: int = 19,
    c_constant: int = 9,
) -> np.ndarray:
    """
    Binarize license plate crop for OCR character segmentation.

    Args:
        image: BGR or Grayscale license plate crop.
        method: 'otsu' for global Otsu binarization, 'adaptive' for local adaptive threshold.
        block_size: Neighborhood block size for adaptive thresholding.
        c_constant: Constant subtracted from mean for adaptive thresholding.

    Returns:
        Binary image (255 for foreground text, 0 for background).
    """
    if image is None or image.size == 0:
        return image

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Pre-denoise
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)

    if method == "adaptive":
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size if block_size % 2 == 1 else block_size + 1,
            c_constant,
        )
    else:
        # Otsu thresholding
        _, binary = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    # Invert if text is darker than background (Indian plates standard is dark text on light background)
    # Check border pixels vs center pixels
    h, w = binary.shape[:2]
    border_pixels = np.concatenate([
        binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]
    ])
    if np.mean(border_pixels) > 128:
        # Border is mostly white (255), meaning inverted polarity
        binary = cv2.bitwise_not(binary)

    return binary


def preprocess_for_ocr(plate_crop: np.ndarray) -> np.ndarray:
    """
    Complete standard preprocessing pipeline for Indian license plates:
    1. Auto-gamma correction for lighting normalization.
    2. CLAHE contrast enhancement.
    3. Bilateral denoising.
    4. Grayscale output with maximized text sharpness.
    """
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop

    # 1. Lighting normalization
    normalized = auto_gamma(plate_crop)

    # 2. CLAHE enhancement
    enhanced = clahe_enhance(normalized, clip_limit=2.5, tile_grid_size=(8, 8))

    # 3. Grayscale conversion
    if len(enhanced.shape) == 3:
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    else:
        gray = enhanced

    # 4. Bilateral filter
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    return denoised
