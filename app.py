"""
app.py
======
Streamlit Web Application for AI-Based Automatic Tooth Segmentation from 2D Dental X-Rays.

Provides:
- Modern clinical interface for uploading panoramic dental radiographs.
- Seamless routing through src/preprocessing.py and src/inference.py.
- Side-by-side visualization of Original X-Ray, Generated Tooth Mask, and Blended Overlay.
- Integrated dataset pairing validator in the sidebar for machine learning data checks.
- One-click synthetic sample generator for immediate testing without external files.
"""

import io
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from src.inference import DummyToothSegmenter
from src.preprocessing import (
    create_mask_overlay,
    preprocess_xray,
    validate_dataset,
)


# --------------------------------------------------------------------------------------
# Page Configuration & Medical / Clinical Styling
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Tooth Segmentation | Dental X-Ray",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished clinical aesthetics
st.markdown(
    """
    <style>
        /* Base styling */
        .main-header {
            font-size: 2.1rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        .badge-sim {
            display: inline-block;
            background-color: #FEF3C7;
            color: #92400E;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 9999px;
            border: 1px solid #FCD34D;
            margin-bottom: 1rem;
        }
        .card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
        .metric-label {
            font-size: 0.82rem;
            color: #64748B;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        .metric-val {
            font-size: 1.35rem;
            font-weight: 700;
            color: #0F172A;
        }
        .stImage > img {
            border-radius: 8px;
            border: 1px solid #CBD5E1;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# Helper: Generate a Realistic Synthetic Dental Panoramic X-Ray for One-Click Demo
# --------------------------------------------------------------------------------------
def generate_sample_panoramic_xray() -> np.ndarray:
    """
    Synthesize an artificial dental panoramic radiograph pattern for instant testing.
    Produces jaw curve and radio-opaque tooth arches with gradient background.
    """
    h, w = 300, 600
    canvas = np.full((h, w), 40, dtype=np.uint8)

    # Gradient mandibular/maxillary bone density background
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    bone_grad = (
        np.sin(x_coords / 70.0) * 15
        + np.sin(y_coords / 40.0) * 20
        + 50
    ).astype(np.uint8)
    canvas = cv2.add(canvas, bone_grad)

    # Draw curved dental arches (upper and lower teeth)
    # Upper arch
    for i in range(14):
        cx = int(80 + i * 32)
        cy = int(120 - 25 * np.sin((i / 13.0) * np.pi))
        cv2.ellipse(canvas, (cx, cy), (10, 18), 0, 0, 360, 210, -1)
        # Enamel cap
        cv2.ellipse(canvas, (cx, cy + 6), (9, 9), 0, 0, 360, 240, -1)

    # Lower arch
    for i in range(14):
        cx = int(85 + i * 31)
        cy = int(185 + 25 * np.sin((i / 13.0) * np.pi))
        cv2.ellipse(canvas, (cx, cy), (10, 18), 0, 0, 360, 205, -1)
        # Enamel cap
        cv2.ellipse(canvas, (cx, cy - 6), (9, 9), 0, 0, 360, 235, -1)

    # Slight blur to emulate radiograph scattering
    canvas = cv2.GaussianBlur(canvas, (5, 5), 1.5)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)


def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    """Convert hex string (e.g. #00FF88) to RGB tuple."""
    hex_code = hex_code.lstrip("#")
    return tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))


# --------------------------------------------------------------------------------------
# Sidebar: Configuration, Model Status & Dataset Validation
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration")

    st.markdown(
        """
        <div style="background-color: #F1F5F9; padding: 10px; border-radius: 8px; font-size: 0.85rem; margin-bottom: 12px;">
            <strong>Inference Engine Status:</strong><br/>
            🟡 <em>Dummy Backend Active</em> (CV Heuristics)<br/>
            <span style="color:#64748B;">Ready to swap with trained PyTorch / TensorFlow weights.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Inference Parameters")
    sensitivity = st.slider(
        "Detection Sensitivity",
        min_value=0.1,
        max_value=1.0,
        value=0.55,
        step=0.05,
        help="Adjusts the thresholding sensitivity for tooth boundary extraction.",
    )

    st.subheader("Visualization Settings")
    overlay_hex = st.color_picker("Mask Overlay Color", "#00FF88")
    overlay_color = hex_to_rgb(overlay_hex)

    alpha_val = st.slider(
        "Mask Opacity (Alpha)",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
        help="Blend transparency of the segmentation mask over the X-ray.",
    )

    st.divider()

    # Expandable Dataset Pairing Validator
    with st.expander("📁 Dataset Integrity Validator", expanded=False):
        st.markdown(
            "<small>Verify that raw dental X-rays have matching ground-truth masks before training.</small>",
            unsafe_allow_html=True,
        )
        img_dir_input = st.text_input("Images Folder", value="data/raw/images")
        mask_dir_input = st.text_input("Masks Folder", value="data/raw/masks")

        if st.button("Run Integrity Check", use_container_width=True):
            with st.spinner("Validating paired files..."):
                val_res = validate_dataset(img_dir_input, mask_dir_input)
                if val_res["is_valid"]:
                    st.success(
                        f"✅ Valid! {val_res['matched_count']} matched image-mask pairs found."
                    )
                else:
                    st.warning(
                        f"⚠️ Status: {val_res['matched_count']} pairs matched, but issues were detected."
                    )
                st.text(val_res["summary"])


# --------------------------------------------------------------------------------------
# Main Application Content
# --------------------------------------------------------------------------------------
st.markdown(
    '<div class="main-header">🦷 AI-Based Automatic Tooth Segmentation</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">2D Panoramic Dental Radiograph Diagnostic & Segmentation Pipeline</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <span class="badge-sim">
        Pipeline Mode: Frontend & Preprocessing Active • Inference: Simulated OpenCV Backend
    </span>
    """,
    unsafe_allow_html=True,
)

with st.expander("ℹ️ About this Architecture & Teammate Integration", expanded=False):
    st.markdown(
        """
        - **Purpose**: This application provides the complete preprocessing and visualization infrastructure for 
          2D panoramic dental X-ray segmentation.
        - **Local Laptop Friendly**: Runs with zero GPU/CUDA dependencies using lightweight libraries.
        - **Deep Learning Handover**: Your teammate can drop trained model weights (`.pth` or `.h5`) 
          directly into `src/inference.py` without rewriting the frontend or data pipelines.
        """
    )

st.divider()

# Input Selection: File Uploader or Demo Sample Button
col_upload, col_sample = st.columns([3, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload a 2D Dental Panoramic X-Ray Image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Only standard 2D panoramic dental radiograph formats are accepted.",
    )

with col_sample:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    use_sample = st.button("🧪 Test with Demo Sample", use_container_width=True)

# Determine image source
image_source = None
source_name = ""

if uploaded_file is not None:
    try:
        # Validate that the file is an actual readable image
        image_source = uploaded_file.getvalue()
        source_name = uploaded_file.name
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
elif use_sample:
    sample_img = generate_sample_panoramic_xray()
    # Encode as JPEG bytes in memory to simulate upload
    _, buffer = cv2.imencode(".png", sample_img)
    image_source = buffer.tobytes()
    source_name = "Synthetic_Panoramic_Sample.png"

# Process and Display if Image Source is available
if image_source is not None:
    try:
        start_time = time.time()

        # Step 1: Preprocessing Module (src/preprocessing.py)
        # Resizes to 256x256 and scales pixel intensities to 0.0 - 1.0
        resized_original, normalized_input = preprocess_xray(
            image_source,
            target_size=(256, 256),
            as_grayscale=False,
        )

        # Step 2: Dummy Inference Module (src/inference.py)
        # Executes OpenCV heuristic segmentation (or DL model when plugged in)
        segmenter = DummyToothSegmenter()
        predicted_mask = segmenter.predict(
            normalized_input,
            threshold_sensitivity=sensitivity,
        )

        # Step 3: Overlay Generation
        # Composites predicted mask over original resized image
        blended_overlay = create_mask_overlay(
            resized_original,
            predicted_mask,
            color=overlay_color,
            alpha=alpha_val,
        )

        elapsed_ms = (time.time() - start_time) * 1000.0

        # Calculate tooth coverage metrics
        total_pixels = predicted_mask.size
        tooth_pixels = np.count_nonzero(predicted_mask)
        tooth_area_pct = (tooth_pixels / total_pixels) * 100.0

        st.success(f"Processing completed in {elapsed_ms:.1f} ms for `{source_name}`")

        # Key Metrics Row
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.markdown(
                f'<div class="card"><div class="metric-label">Input Resolution</div>'
                f'<div class="metric-val">256 × 256</div></div>',
                unsafe_allow_html=True,
            )
        with metric_col2:
            st.markdown(
                f'<div class="card"><div class="metric-label">Pixel Normalization</div>'
                f'<div class="metric-val">[0.0, 1.0]</div></div>',
                unsafe_allow_html=True,
            )
        with metric_col3:
            st.markdown(
                f'<div class="card"><div class="metric-label">Segmented Area</div>'
                f'<div class="metric-val">{tooth_area_pct:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
        with metric_col4:
            st.markdown(
                f'<div class="card"><div class="metric-label">Inference Time</div>'
                f'<div class="metric-val">{elapsed_ms:.1f} ms</div></div>',
                unsafe_allow_html=True,
            )

        # Results Side-by-Side Display
        st.subheader("Segmentation Results Visualizer")
        col_img1, col_img2, col_img3 = st.columns(3)

        with col_img1:
            st.markdown("##### 1. Original X-Ray (256×256)")
            st.image(
                resized_original,
                caption="Normalized Preprocessed Input",
                use_container_width=True,
            )

        with col_img2:
            st.markdown("##### 2. Predicted Tooth Mask")
            st.image(
                predicted_mask,
                caption=f"Binary Mask (0: Bone/Back, 255: Tooth)",
                use_container_width=True,
            )

        with col_img3:
            st.markdown("##### 3. Blended Overlay")
            st.image(
                blended_overlay,
                caption=f"Mask Overlay (Opacity: {int(alpha_val*100)}%)",
                use_container_width=True,
            )

        # Download Actions
        st.divider()
        st.markdown("##### 💾 Export Results")
        d_col1, d_col2 = st.columns(2)

        # Encode Mask to PNG bytes
        is_success_mask, mask_buffer = cv2.imencode(".png", predicted_mask)
        if is_success_mask:
            with d_col1:
                st.download_button(
                    label="⬇️ Download Predicted Mask (.PNG)",
                    data=mask_buffer.tobytes(),
                    file_name="tooth_segmentation_mask.png",
                    mime="image/png",
                    use_container_width=True,
                )

        # Encode Blended Overlay to PNG bytes
        overlay_bgr = cv2.cvtColor(blended_overlay, cv2.COLOR_RGB2BGR)
        is_success_overlay, overlay_buffer = cv2.imencode(".png", overlay_bgr)
        if is_success_overlay:
            with d_col2:
                st.download_button(
                    label="⬇️ Download Blended Overlay (.PNG)",
                    data=overlay_buffer.tobytes(),
                    file_name="tooth_segmentation_overlay.png",
                    mime="image/png",
                    use_container_width=True,
                )

    except ValueError as ve:
        st.error(f"Image Format Error: {ve}")
    except Exception as e:
        st.error(f"Unexpected processing error occurred: {e}")

else:
    st.info("👆 Upload a 2D panoramic X-ray image above or click **'Test with Demo Sample'** to run the pipeline.")
