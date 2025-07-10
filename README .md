# Face Recognition Module for Smart Mirror

This module performs face embedding extraction for identity matching. It is designed to work downstream of a face detection module and serves as the recognition core of the smart mirror system.

## Overview

- Lightweight ONNX-based face recognition
- Uses MTCNN for face alignment and EdgeFace for embedding extraction
- Optimized for Jetson Nano deployment (supports TensorRT)
- Validated on Windows (x64) and Jetson Nano (ARM64, CUDA 11.4+)

## Input / Output Specifications

| Item            | Description                                      | Notes                              |
|-----------------|--------------------------------------------------|------------------------------------|
| Input Type      | RGB face image (file path or image array)        | `PIL.Image`, `np.ndarray`, or path |
| Input Size      | Detected face region resized and aligned to 112×112 via MTCNN | |
| Input Format    | Normalized float32 tensor: (1, 3, 112, 112), range [-1, 1], mean=0.5 std=0.5 | CHW format |
| Output Type     | 512-dimensional identity embedding               | `np.ndarray` shape `(1, 512)`      |
| ONNX Input Name | `"input"` (default)                              | Use `session.get_inputs()[0].name` to confirm |
| Alignment Tool  | `face_alignment/align.py`                        | Uses MTCNN                         |

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/face_recognition_jetson.git
cd face_recognition_jetson
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** Use Python 3.8–3.10 and ensure CUDA is available if on Jetson.

### 3. Run ONNX Inference (Example)

```bash
python scripts/inference_onnx.py
```

This script loads an image, aligns the face, and prints a 512-dim embedding vector.

## TensorRT Conversion (Jetson Only)

```bash
# Jetson only (TensorRT installed):
bash scripts/build_trt_engine.sh
```

This generates a `.trt` engine from ONNX for faster inference using TensorRT.

## Usage Example (API Style)

```python
from face_alignment.align import get_aligned_face
import onnxruntime as ort
import cv2
import numpy as np

aligned = get_aligned_face("test.jpg")

# preprocess: resize to 112x112, RGB, normalize, transpose, batch
img = cv2.resize(aligned, (112, 112))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
img = (img - 0.5) / 0.5
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)

session = ort.InferenceSession("onnx/edgeface_xs_gamma_06.onnx")
embedding = session.run(None, {"input": img})[0]
```

## Model Details

- Name: `EdgeFace-xs (γ=0.6)`
- Size: ~1.9MB
- Format: PyTorch `.pt` + ONNX `.onnx`
- Embedding Size: 512
- Source: [otroshi/edgeface](https://github.com/otroshi/edgeface)

## Team Integration Guide

| Module        | Interface Format                              |
|---------------|------------------------------------------------|
| Detection     | Output should be RGB image (BGR also supported) |
| Alignment     | Auto handled by `face_alignment/align.py`     |
| Recognition   | Outputs 512D embedding for matching            |

## Contributors

- Zixuan Xu (@ars20tot)  
- Maintained as part of the *Smart Mirror with Personalized Insights* project