import scipy.io
import argparse
import csv
from tqdm import tqdm
from random import randint
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="The MATLAB file to read and extract GPS coordinates from", required=True, type=str)
    parser.add_argument("--output", help="The output folder", required=True, type=str)
    return parser.parse_args()

def main():
    args = get_args()
    
    train_files = []
    val_files = []
    
    mat = scipy.io.loadmat('GPS_Long_Lat_Compass.mat')
    coords = mat['GPS_Long_Lat_Compass']
    
    for coord_index in tqdm(range(len(coords))):
        lat, lon = coords[coord_index][0], coords[coord_index][1]
        
        # Set to training or testing data
        if -76 <= lon <= 70: 
            if randint(0, 9) == 0:
                for i in range(6):
                    val_files.append([f'images/{str(coord_index + 1).zfill(6)}_{i}.jpg', [lat, lon]])
            else:
                for i in range(5):
                    train_files.append([f'images/{str(coord_index + 1).zfill(6)}_{i}.jpg', [lat, lon]])
        
    with open(f'{args.output}/train_data.csv', mode='w', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerows(train_files)
    
    with open(f'{args.output}/val_data.csv', mode='w', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerows(train_files)
        
    print('Train Files:', len(train_files))
    print('Val Files:', len(val_files))
    print('Total Files:', len(train_files) + len(val_files))
    print('Train Files to Val Files Ratio:', len(train_files) / len(val_files))

if __name__ == '__main__':
    main()
