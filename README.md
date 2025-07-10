# Face Recognition Module for Smart Mirror

This repository implements a Jetson Nano-compatible lightweight face recognition system based on the EdgeFace model. It is part of the **Smart Mirror with Personalized Insights** project.

## Overview

- Lightweight ONNX-based face recognition
- Uses MTCNN for face alignment and EdgeFace for embedding extraction
- Optimized for Jetson Nano deployment (supports TensorRT)
- Tested on both Windows and Jetson environments

## Input / Output Specifications

| Item            | Description                                | Notes                              |
|-----------------|--------------------------------------------|------------------------------------|
| Input Type      | RGB face image (file path or image array)  | `PIL.Image`, `np.ndarray`, or path |
| Input Size      | Auto-detected face aligned to `112×112`    | via MTCNN                          |
| Input Format    | `(1, 3, 112, 112)` float32, range [-1, 1]  | CHW format                         |
| Output Type     | 512-dimensional identity embedding          | `np.ndarray` shape `(1, 512)`      |
| ONNX Input Name | `"input"`                                   | Required for session.run()         |
| Alignment Tool  | `face_alignment/align.py`                   | Uses MTCNN                         |

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ars20tot/face_recognition_jetson.git
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
bash scripts/build_trt_engine.sh
```

This generates a `.trt` engine from ONNX for faster inference using TensorRT.

## Usage Example (API Style)

```python
from face_alignment.align import get_aligned_face
import onnxruntime as ort

aligned = get_aligned_face("test.jpg")
img_tensor = preprocess(aligned)
session = ort.InferenceSession("onnx/edgeface_xs_gamma_06.onnx")
embedding = session.run(None, {"input": img_tensor})[0]
```

## Model Details

- Name: `EdgeFace-xs (γ=0.6)`
- Size: ~1.9MB
- Format: PyTorch `.pt` + ONNX `.onnx`
- Embedding Size: 512
- Source: [otroshi/edgeface](https://github.com/otroshi/edgeface)

## Team Integration Guide

| Module        | Interface Format                    |
|---------------|--------------------------------------|
| Detection     | Output should be RGB image (BGR also supported) |
| Alignment     | Auto handled by `face_alignment/align.py` |
| Recognition   | Outputs 512D embedding for matching  |

## Contributors

- Zixuan Xu (@ars20tot)
- Part of the Smart Mirror with Personalized Insights project
