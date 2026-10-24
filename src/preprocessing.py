"""
src/preprocessing.py
====================
Data Preprocessing Module for 2D Panoramic Dental X-Ray Images.

Responsibilities:
1. Robust OpenCV-based loading from file paths or in-memory byte buffers (Streamlit).
2. Resizing images to standard target dimensions (256x256) and normalizing pixels (0-255 -> 0.0-1.0).
3. Visual utilities for denormalization and mask-over-image blending.
4. Dataset integrity validation: verifies that every input X-ray image has a matching ground-truth mask.
"""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


# Standard default target size for dental segmentation models (e.g., U-Net)
DEFAULT_TARGET_SIZE = (256, 256)

# Supported image formats for dental radiography
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_image(
    source: Union[str, Path, bytes, io.BytesIO],
    as_grayscale: bool = False,
) -> np.ndarray:
    """
    Load an X-ray image from either a file path on disk or an in-memory byte buffer.

    Parameters
    ----------
    source : Union[str, Path, bytes, io.BytesIO]
        File path (str or Path) or binary buffer from Streamlit file uploader.
    as_grayscale : bool, default=False
        If True, loads image as single-channel grayscale (cv2.IMREAD_GRAYSCALE).
        If False, loads image and converts from OpenCV's BGR to RGB.

    Returns
    -------
    np.ndarray
        Loaded image array in uint8 format (shape: [H, W] or [H, W, 3]).

    Raises
    ------
    FileNotFoundError
        If a provided file path does not exist on disk.
    ValueError
        If the image cannot be decoded or is corrupted.
    """
    flag = cv2.IMREAD_GRAYSCALE if as_grayscale else cv2.IMREAD_COLOR

    if isinstance(source, (bytes, bytearray)):
        file_bytes = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(file_bytes, flag)
    elif isinstance(source, io.BytesIO):
        file_bytes = np.frombuffer(source.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, flag)
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path.resolve()}")
        # cv2.imread does not handle non-ASCII paths well on some Windows systems;
        # using imdecode with fromfile ensures robust cross-platform path handling.
        img_array = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(img_array, flag)

    if img is None or img.size == 0:
        raise ValueError("Failed to decode image. File may be corrupted or unsupported.")

    # Convert BGR to RGB for standard color representation if not grayscale
    if not as_grayscale and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def resize_and_normalize(
    image: np.ndarray,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize image to a fixed resolution and scale pixel intensities from [0, 255] to [0.0, 1.0].

    Parameters
    ----------
    image : np.ndarray
        Input image array in uint8 format (0-255).
    target_size : Tuple[int, int], default=(256, 256)
        Target (width, height) resolution for model input.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - resized_uint8: uint8 image resized to target_size (values 0-255).
        - normalized_float32: float32 image normalized to [0.0, 1.0].
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected image as np.ndarray, received {type(image)}")

    # Use INTER_AREA for downsizing (crisper details, reduces aliasing)
    # Use INTER_LINEAR for upsizing
    h, w = image.shape[:2]
    interpolation = (
        cv2.INTER_AREA
        if (w >= target_size[0] and h >= target_size[1])
        else cv2.INTER_LINEAR
    )

    resized_uint8 = cv2.resize(image, target_size, interpolation=interpolation)
    normalized_float32 = resized_uint8.astype(np.float32) / 255.0

    return resized_uint8, normalized_float32


def preprocess_xray(
    source: Union[str, Path, bytes, io.BytesIO],
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    as_grayscale: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    High-level end-to-end preprocessing pipeline for a single 2D dental X-ray.

    Pipeline:
        1. Load raw image from path or in-memory byte stream.
        2. Resize to target dimension (e.g. 256x256).
        3. Normalize pixel intensities to range [0.0, 1.0] (float32).

    Parameters
    ----------
    source : Union[str, Path, bytes, io.BytesIO]
        Input image source.
    target_size : Tuple[int, int], default=(256, 256)
        Output dimensions (width, height).
    as_grayscale : bool, default=False
        Whether to convert the image to single-channel grayscale.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - original_resized_uint8: Resized uint8 image (0-255) for UI rendering.
        - normalized_float32: Normalized float32 image (0.0-1.0) ready for model inference.
    """
    raw_img = load_image(source, as_grayscale=as_grayscale)
    return resize_and_normalize(raw_img, target_size=target_size)


def denormalize_image(normalized: np.ndarray) -> np.ndarray:
    """
    Convert a normalized float32 image ([0.0, 1.0]) back to uint8 ([0, 255]).

    Parameters
    ----------
    normalized : np.ndarray
        Array with values in [0.0, 1.0].

    Returns
    -------
    np.ndarray
        Array with values clipped to [0, 255] in uint8 format.
    """
    clipped = np.clip(normalized * 255.0, 0, 255)
    return clipped.astype(np.uint8)


def create_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 128),
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Generate a blended composite of a binary segmentation mask overlaid on an X-ray.

    Parameters
    ----------
    image : np.ndarray
        Original image (uint8, either grayscale [H, W] or RGB [H, W, 3]).
    mask : np.ndarray
        Binary segmentation mask (values 0 or 255, or boolean [H, W]).
    color : Tuple[int, int, int], default=(0, 255, 128)
        RGB color tuple for the overlay (default: luminous medical green).
    alpha : float, default=0.45
        Opacity of the mask overlay (0.0 = completely transparent, 1.0 = solid).

    Returns
    -------
    np.ndarray
        Blended RGB image of shape [H, W, 3] in uint8 format.
    """
    # Ensure base image is RGB uint8
    if image.dtype != np.uint8:
        base_img = denormalize_image(image)
    else:
        base_img = image.copy()

    if len(base_img.shape) == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
    elif base_img.shape[2] == 4:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_RGBA2RGB)

    # Ensure mask is 2D binary uint8 (0 or 255)
    binary_mask = (mask > 0).astype(np.uint8) * 255
    if len(binary_mask.shape) == 3:
        binary_mask = binary_mask[:, :, 0]

    # Create solid color canvas of same dimensions
    colored_mask = np.zeros_like(base_img)
    colored_mask[binary_mask > 0] = color

    # Blend original and colored mask using weighted addition
    blended = cv2.addWeighted(colored_mask, alpha, base_img, 1.0, 0.0)

    # Keep non-masked areas completely crisp from the original image
    result = np.where(binary_mask[:, :, None] > 0, blended, base_img)
    return result


def validate_dataset(
    images_dir: Union[str, Path],
    masks_dir: Union[str, Path],
    valid_extensions: Optional[set] = None,
    verify_dimensions: bool = False,
) -> Dict[str, Any]:
    """
    Verify dataset pairing between input dental X-rays and ground-truth segmentation masks.

    Checks:
    1. Both directories exist and are accessible.
    2. Every image has an identical or matching stem mask file (e.g., `001.png` or `001_mask.png`).
    3. Identifies orphan masks (masks without an image).
    4. (Optional) Verifies matching spatial dimensions (height, width).

    Parameters
    ----------
    images_dir : Union[str, Path]
        Path to directory containing input X-ray images.
    masks_dir : Union[str, Path]
        Path to directory containing ground-truth segmentation masks.
    valid_extensions : Optional[set], default=None
        Set of valid file extensions (defaults to VALID_IMAGE_EXTENSIONS).
    verify_dimensions : bool, default=False
        If True, loads image headers to ensure image and mask have matching (H, W).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "is_valid": bool, True if all images have paired masks and no critical errors.
        - "total_images": int, total count of valid images detected.
        - "total_masks": int, total count of valid masks detected.
        - "matched_count": int, count of paired images and masks.
        - "matched_pairs": List[Tuple[Path, Path]], list of (image_path, mask_path).
        - "missing_masks": List[Path], image paths lacking a matching mask.
        - "orphan_masks": List[Path], mask paths lacking a matching image.
        - "dimension_mismatches": List[Tuple[Path, Path, Tuple, Tuple]], dimension conflicts.
        - "summary": str, human-readable summary of validation status.
    """
    exts = valid_extensions or VALID_IMAGE_EXTENSIONS
    img_dir_path = Path(images_dir)
    mask_dir_path = Path(masks_dir)

    results: Dict[str, Any] = {
        "is_valid": False,
        "total_images": 0,
        "total_masks": 0,
        "matched_count": 0,
        "matched_pairs": [],
        "missing_masks": [],
        "orphan_masks": [],
        "dimension_mismatches": [],
        "summary": "",
    }

    if not img_dir_path.exists() or not img_dir_path.is_dir():
        results["summary"] = f"Images directory does not exist: {img_dir_path.resolve()}"
        return results

    if not mask_dir_path.exists() or not mask_dir_path.is_dir():
        results["summary"] = f"Masks directory does not exist: {mask_dir_path.resolve()}"
        return results

    # Index all mask files by stem variations for fast O(1) matching
    mask_files = [
        f for f in mask_dir_path.iterdir() if f.is_file() and f.suffix.lower() in exts
    ]
    results["total_masks"] = len(mask_files)

    # Map possible lookup keys to actual mask path
    # Handles: 'image01' -> 'image01.png', 'image01_mask.png', 'image01_gt.png'
    mask_lookup: Dict[str, Path] = {}
    for mf in mask_files:
        stem = mf.stem.lower()
        mask_lookup[stem] = mf
        # Strip common mask suffixes for flexible matching
        clean_stem = (
            stem.removesuffix("_mask")
            .removesuffix("-mask")
            .removesuffix("_gt")
            .removesuffix("_seg")
        )
        mask_lookup[clean_stem] = mf

    # Scan and match input images
    image_files = [
        f for f in img_dir_path.iterdir() if f.is_file() and f.suffix.lower() in exts
    ]
    results["total_images"] = len(image_files)

    if results["total_images"] == 0:
        results["summary"] = f"No valid images found in {img_dir_path.resolve()}."
        return results

    matched_mask_paths = set()

    for img_path in sorted(image_files):
        img_stem = img_path.stem.lower()
        matched_mask = mask_lookup.get(img_stem)

        if matched_mask is not None:
            results["matched_pairs"].append((img_path, matched_mask))
            matched_mask_paths.add(matched_mask)

            if verify_dimensions:
                try:
                    img_shape = load_image(img_path).shape[:2]
                    mask_shape = load_image(matched_mask).shape[:2]
                    if img_shape != mask_shape:
                        results["dimension_mismatches"].append(
                            (img_path, matched_mask, img_shape, mask_shape)
                        )
                except Exception as e:
                    results["summary"] += f"\nError reading dimensions for {img_path.name}: {e}"
        else:
            results["missing_masks"].append(img_path)

    results["matched_count"] = len(results["matched_pairs"])

    # Find orphan masks (masks that have no corresponding image)
    for mf in mask_files:
        if mf not in matched_mask_paths:
            results["orphan_masks"].append(mf)

    # Validation criteria:
    # 1. At least one image is found.
    # 2. No image is missing a ground truth mask.
    # 3. No dimension mismatches (if dimension check was requested).
    has_missing = len(results["missing_masks"]) > 0
    has_dim_mismatch = len(results["dimension_mismatches"]) > 0
    results["is_valid"] = (results["total_images"] > 0) and (not has_missing) and (not has_dim_mismatch)

    summary_lines = [
        f"Validation Result: {'PASSED (Ready for training)' if results['is_valid'] else 'FAILED'}",
        f"Total Images Found: {results['total_images']}",
        f"Total Masks Found:  {results['total_masks']}",
        f"Successfully Paired: {results['matched_count']}",
        f"Missing Masks:      {len(results['missing_masks'])}",
        f"Orphan Masks:       {len(results['orphan_masks'])}",
    ]
    if verify_dimensions:
        summary_lines.append(f"Dimension Mismatches: {len(results['dimension_mismatches'])}")

    results["summary"] = "\n".join(summary_lines)
    return results


if __name__ == "__main__":
    print("Testing Preprocessing Module...")
    # Generate a dummy synthetic 2D dental X-ray array for self-test
    synthetic_xray = np.random.randint(50, 220, size=(512, 1024, 3), dtype=np.uint8)
    resized, normalized = resize_and_normalize(synthetic_xray, target_size=(256, 256))

    print(f"Original shape: {synthetic_xray.shape}")
    print(f"Resized shape:  {resized.shape}, dtype: {resized.dtype}, range: [{resized.min()}, {resized.max()}]")
    print(f"Normalized shape: {normalized.shape}, dtype: {normalized.dtype}, range: [{normalized.min():.2f}, {normalized.max():.2f}]")
    print("Preprocessing self-test passed successfully.")
