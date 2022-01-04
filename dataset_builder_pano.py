import scipy.io
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from random import randint
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="The MATLAB file to read and extract GPS coordinates from", required=True, type=str)
    parser.add_argument("--images", help="The path to the images folder, (defaults to: images/)", default='images/', type=str)
    parser.add_argument("--output", help="The output folder", required=True, type=str)
    return parser.parse_args()

args = get_args()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

def multi_label(num, decimals=4):
    if decimals > 0:
      num = str(round(num * (10 ** decimals)))
    else:
      num = str(round(num))
    
    if num[0] == '-':
        label = np.array([1])
        num = num[1:]
    else:
        label = np.array([0])
    
    for digit in num:
        label = np.concatenate((label, np.eye(10)[int(digit)]))
    
    return label

def get_data(coord, coord_index):
    lat, lon = coord[0], coord[1]
    img_arr = torch.tensor([])
    
    for i in range(5):
        img_path = os.path.join(args.images, f'{str(coord_index + 1).zfill(6)}_{i}.jpg')
        img = Image.open(img_path)
        img = transform(img)
        img_arr = torch.cat((img_arr, img), 2)
    
    lat_multi_label = multi_label(lat)
    lon_multi_label = multi_label(lon)
    
    target = np.concatenate((lat_multi_label, lon_multi_label))
    
    return [img_arr, target]

def main():
    mat = scipy.io.loadmat('GPS_Long_Lat_Compass.mat')
    coords = mat['GPS_Compass']
    
    train_data_path = os.path.join(args.output, 'train')
    os.makedirs(train_data_path, exist_ok=True)
    val_data_path = os.path.join(args.output, 'val')
    os.makedirs(val_data_path, exist_ok=True)
    
    val_count = 0
    train_count = 0
    
    for coord_index in tqdm(range(len(coords))):
        coord = coords[coord_index]
        lon = coord[1]
        # Set to training or testing data
        if -76 <= lon <= -70:
            if randint(0, 9) == 0:
                data = get_data(coord, coord_index)
                val_data_path = os.path.join(args.output, f'val/street_view_{coord_index}.npy')
                np.save(val_data_path, data)
                val_count += 1
            else:
                data = get_data(coord, coord_index)
                train_data_path = os.path.join(args.output, f'train/street_view_{coord_index}.npy')
                np.save(train_data_path, data)
                train_count += 1
    
    print('Train Files:', train_count)
    print('Val Files:', val_count)

if __name__ == '__main__':
    main()
