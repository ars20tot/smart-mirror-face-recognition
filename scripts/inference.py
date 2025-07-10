import os
import sys
import torch
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from face_alignment import align
from models import get_model

model_name = "edgeface_xs_gamma_06"
model = get_model(model_name)
checkpoint_path = f'checkpoints/{model_name}.pt'

state_dict = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

path = 'test_data/jf.jpg'
aligned = align.get_aligned_face(path)
input_tensor = transform(aligned).unsqueeze(0)

with torch.no_grad():
    embedding = model(input_tensor)

print(embedding.shape)
print(embedding)


