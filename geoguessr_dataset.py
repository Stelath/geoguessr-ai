from torch.utils.data import Dataset
import numpy as np
import os, os.path

class GeoGuessrDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __len__(self):
        return len([name for name in os.listdir(self.data_dir) if os.path.isfile(name)])

    def __getitem__(self, idx):
        # Load numpy array from index of listdir
        data_path = os.join(self.data_dir, os.listdir(self.data_dir)[idx])
        data = np.load(data_path)
        return data[0], data[1]