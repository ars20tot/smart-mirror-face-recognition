import sys
import os
import onnxruntime as ort
import numpy as np
import cv2


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from face_alignment.align import get_aligned_face

session = ort.InferenceSession("onnx/edgeface_xs_gamma_06.onnx", providers=["CPUExecutionProvider"])

path = "test_data/jf.jpg"
from PIL import Image
aligned_pil = get_aligned_face(path)
aligned = cv2.cvtColor(np.array(aligned_pil), cv2.COLOR_RGB2BGR)


if aligned is None:
    print(" Face alignment failed.")
    exit()

print(f"[DEBUG] aligned type: {type(aligned)}, shape: {aligned.shape}")

img = cv2.resize(aligned, (112, 112))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
img = (img - 0.5) / 0.5
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)

embedding = session.run(None, {"input": img})[0]
print(embedding.shape)
print(embedding)
