import torch
from torchvision import transforms
from models import get_model
from face_alignment import align
import numpy as np

model_name = "edgeface_xs_gamma_06"
checkpoint_path = f"checkpoints/{model_name}.pt"
model = get_model(model_name)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def extract_embedding(image_path):
    aligned = align.get_aligned_face(image_path)
    img = transform(aligned).unsqueeze(0)
    embedding = model(img).detach().numpy()
    return embedding

def compare_embeddings(emb1, emb2):
    return np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def is_match(emb1, emb2, threshold=0.5):
    score = compare_embeddings(emb1, emb2)
    return score >= threshold

