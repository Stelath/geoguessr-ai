import scipy.io
import argparse
import csv
import torchvision.transforms as transforms
from torchvision.io import read_image
from tqdm import tqdm
from random import randint
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="The MATLAB file to read and extract GPS coordinates from", required=True, type=str)
    parser.add_argument("--images", help="The path to the images folder, (defaults to: /images)", default='/images', type=str)
    parser.add_argument("--output", help="The output folder", required=True, type=str)
    return parser.parse_args()

# def main():
#     args = get_args()
    
#     train_files = []
#     val_files = []
    
#     mat = scipy.io.loadmat('GPS_Long_Lat_Compass.mat')
#     coords = mat['GPS_Long_Lat_Compass']
    
#     for coord_index in tqdm(range(len(coords))):
#         lat, lon = coords[coord_index][0], coords[coord_index][1]
        
#         # Set to training or testing data
#         if -76 <= lon <= 70: 
#             if randint(0, 9) == 0:
#                 for i in range(6):
#                     val_files.append([f'images/{str(coord_index + 1).zfill(6)}_{i}.jpg', [lat, lon]])
#             else:
#                 for i in range(5):
#                     train_files.append([f'images/{str(coord_index + 1).zfill(6)}_{i}.jpg', [lat, lon]])
        
#     with open(f'{args.output}/train_data.csv', mode='w', encoding='utf8') as f:
#         writer = csv.writer(f)
#         writer.writerows(train_files)
    
#     with open(f'{args.output}/val_data.csv', mode='w', encoding='utf8') as f:
#         writer = csv.writer(f)
#         writer.writerows(train_files)
        
#     print('Train Files:', len(train_files))
#     print('Val Files:', len(val_files))
#     print('Total Files:', len(train_files) + len(val_files))
#     print('Train Files to Val Files Ratio:', len(train_files) / len(val_files))

args = get_args()
    
train_data = []
val_data = []

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

def get_data(coord, coord_index):
    lat, lon = coord[0], coord[1]
    img_arr = []
    channels = tuple(img_arr_r_channel = [], img_arr_g_channel = [], img_arr_b_channel = [])
    for i in range(6):
        img = read_image(os.path.join(args.images, f'{str(coord_index + 1).zfill(6)}_{i}.jpg'))
        img = transform(img)
        
        for channel in channels:
            channel.append(img)
    
    for channel in channels:
        img_arr.append(channel)
        
    return [img_arr, [lat, lon]]

def main():
    mat = scipy.io.loadmat('GPS_Long_Lat_Compass.mat')
    coords = mat['GPS_Long_Lat_Compass']
    
    val_count = 0
    train_count = 0
    
    for coord_index, coord in tqdm(enumerate(coords)):
        lon = coord[0]
        # Set to training or testing data
        if -76 <= lon <= 70: 
            if randint(0, 9) == 0:
                data = get_data(coord, coord_index)
                val_data.append(data)
                val_count += 1
            else:
                data = get_data(coord, coord_index)
                train_data.append(data)
                train_count += 1
    
    train_data_path = os.path.join(args.output, 'training_data.npy')
    np.random.shuffle(train_data)
    np.save(train_data_path, training_data)
    
    val_data_path = os.path.join(args.output, 'training_data.npy')
    np.random.shuffle(val_data)
    np.save(val_data_path, val_data)
    
    print('Train Files:', train_count)
    print('Val Files:', val_count)

if __name__ == '__main__':
    main()
