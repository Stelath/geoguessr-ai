import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os, os.path
from PIL import Image

class GeoGuessrDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.targets = np.load(os.path.join(data_dir, 'targets.npy'), allow_pickle=True)

    def __len__(self):
        return len(os.listdir(self.data_dir)) - 1

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, f'street_view_{idx}.jpg')
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                normalize,
            ])
        
        img = pil_loader(data_path)
        data = transform(img)
        
        target = torch.tensor(self.targets[idx], dtype=torch.float)
        
        return data, target
    
def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")