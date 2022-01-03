import torch
from torch.utils.data import Dataset
import numpy as np
import os, os.path

class GeoGuessrDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        # Load numpy array from index of listdir
        data_path = os.path.join(self.data_dir, os.listdir(self.data_dir)[idx])
        data = np.load(data_path, allow_pickle=True)
        
        target = torch.tensor(data[1], dtype=torch.float)
        data = data[0]
        
        return data, target