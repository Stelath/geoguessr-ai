import torch

def round_tensor(tensor, decimals=4):
    return torch.round(tensor * 10 ** decimals) / (10 ** decimals)