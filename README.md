# 🦷 AI-Based Automatic Tooth Segmentation from 2D Dental X-Ray Images

A modular, lightweight computer vision infrastructure and web application for automatic tooth segmentation from 2D panoramic dental radiographs.

This repository provides:
- **Frontend**: Clinical-grade Streamlit web interface for uploading radiographs, adjusting parameters, and viewing side-by-side masks and overlays.
- **Data Preprocessing Pipeline (`src/preprocessing.py`)**: Robust OpenCV-based loading, resizing to `256x256`, pixel normalization (`[0.0, 1.0]`), mask-over-radiograph blending, and dataset validation.
- **Simulated Inference Backend (`src/inference.py`)**: Classical computer vision heuristics (CLAHE, bilateral denoising, adaptive thresholding, and morphological filtering) to simulate AI predictions locally on low-spec hardware without needing GPUs or heavy deep learning frameworks.
- **Teammate Handover Architecture**: Pre-configured integration points for the deep learning engineer to drop in trained PyTorch (`.pth`) or TensorFlow/Keras (`.h5`) model weights.

---

## 📂 Project Architecture

```text
tooth_segmentation/
├── app.py                      # Streamlit interactive frontend application
├── requirements.txt            # Lightweight dependencies (No PyTorch/TensorFlow)
├── README.md                   # Setup guide and developer instructions
├── .gitignore                  # Git exclusions for checkpoints, venvs, and cache
├── src/
│   ├── __init__.py             # Module exports
│   ├── preprocessing.py        # Loading, resizing, normalization & dataset validation
│   └── inference.py            # Simulated AI inference & DL model handover stubs
└── data/
    ├── raw/
    │   ├── images/             # Raw 2D dental X-rays (.png, .jpg)
    │   └── masks/              # Ground truth binary tooth masks (.png)
    └── processed/              # Preprocessed 256x256 training pairs
```

---

## 🚀 Step-by-Step Setup & Run Instructions

### Step 1: Open a Terminal in the Project Root
Ensure your current directory is `tooth_segmentation`:
```powershell
cd c:\Projects\tooth_segmentation
```

### Step 2: Create a Python Virtual Environment
Keep your local environment clean and isolated:

- **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
  *(If execution policies prevent running scripts, execute: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

- **On Windows (Command Prompt):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate.bat
  ```

- **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 3: Upgrade pip and Install Lightweight Dependencies
Install only the lightweight frontend and computer vision packages (keeps installation fast and laptop resource usage minimal):
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Launch the Streamlit Web Application
Run the local development server:
```bash
streamlit run app.py
```
Streamlit will automatically open your browser at `http://localhost:8501`.

---

## 🧪 Testing the Pipeline Without Real Data

You can immediately test the pipeline without downloading any dental radiographs:
1. Launch the app (`streamlit run app.py`).
2. Click the **"🧪 Test with Demo Sample"** button.
3. The app synthesizes an artificial panoramic radiograph, runs it through `src/preprocessing.py` and `src/inference.py`, and displays:
   - **Original X-Ray (256×256)**
   - **Predicted Tooth Mask (Binary)**
   - **Blended Overlay (Mask composited over radiograph)**
4. Download the generated mask or overlay using the export buttons.

---

## 📁 Dataset Integrity Validation

Before training deep learning models, verify that every image has an associated ground-truth mask:

### Option A: From the Streamlit Web Interface
1. In the sidebar, expand **"📁 Dataset Integrity Validator"**.
2. Enter your image and mask folders (defaults to `data/raw/images` and `data/raw/masks`).
3. Click **"Run Integrity Check"** to view a live report.

### Option B: From Python CLI
Run the self-test built into `preprocessing.py`:
```bash
python -m src.preprocessing
```
Or run custom script checks:
```python
from src.preprocessing import validate_dataset

report = validate_dataset("data/raw/images", "data/raw/masks", verify_dimensions=True)
print(report["summary"])
```

---

## 🤝 Teammate Handover: Connecting PyTorch / TensorFlow

When your teammate finishes training their deep learning model (e.g. U-Net, Mask R-CNN, ViT), they can plug it into the application in 3 steps:

1. **Install Model Framework** on their environment:
   ```bash
   pip install torch torchvision
   # OR
   pip install tensorflow
   ```

2. **Open `src/inference.py`**:
   - Locate the marked sections: `[TODO: TEAMMATE - STEP 1]`, `[STEP 2]`, and `[STEP 3]`.
   - Uncomment the PyTorch or TensorFlow imports.
   - Point `model_weights_path` to their trained `.pth` or `.h5` file.
   - Enable the tensor conversion and forward pass.

3. **Input / Output Contract**:
   - **Input**: `preprocessed_image` (shape: `[256, 256]` or `[256, 256, 3]`, values `0.0 - 1.0`).
   - **Output**: `binary_mask` (shape: `[256, 256]`, dtype `uint8`, values `0` and `255`).

No modifications to `app.py` or `preprocessing.py` are required!

---

## 📤 Pushing to GitHub

Initialize git, commit your files, and push to your GitHub repository:
```bash
git init
git add .
git commit -m "feat: complete frontend, preprocessing pipeline, and dummy inference backend"
git branch -M main
git remote add origin https://github.com/<your-username>/tooth-segmentation.git
git push -u origin main
```
