import torch
from torchvision import transforms
from face_alignment import align
from models import get_model

model_name = "edgeface_s_gamma_06"
model = get_model(model_name)
checkpoint_path = f'checkpoints/{model_name}.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')).eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

path = 'jf.jpg'
aligned = align.get_aligned_face(path)
input_tensor = transform(aligned).unsqueeze(0)

with torch.no_grad():
    embedding = model(input_tensor)

print(embedding.shape)


