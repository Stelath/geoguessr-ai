import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import csv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset1", help="The first dataset folder", required=True, type=str)
    parser.add_argument("--dataset2", help="The second dataset folder", type=str)
    parser.add_argument("--new_dataset", help="The new dataset folder to be created", type=str)
    return parser.parse_args()

args = get_args()

def main():
    with open(os.path.join(args.dataset1, 'picture_coords.csv'), newline='') as f:
        reader = csv.reader(f)
        data1 = list(reader)
    with open(os.path.join(args.dataset2, 'picture_coords.csv'), newline='') as f:
        reader = csv.reader(f)
        data2 = list(reader)
    
    coord_output_file = open(os.path.join(args.new_dataset, 'picture_coords.csv'), 'w', newline='')
    csv_writer = csv.writer(coord_output_file)
    
    os.makedirs(args.new_dataset, exist_ok=True)
    
    failed_images = 0
    for img_num in tqdm(range(len(os.listdir(args.dataset1)) - 1)):
        img_num_new = img_num - failed_images
        img = cv2.imread(os.path.join(args.dataset1, 'street_view_' + str(img_num) + '.jpg'))
        cropped_image = img[0:200, 0:200]
        if not all((i == (223, 227, 228)).all() for i in cropped_image):
            cv2.imwrite(os.path.join(args.new_dataset, ('street_view_' + str(img_num_new) + '.jpg')), img)
            csv_writer.writerow(data1[img_num])
        else:
            failed_images += 1
    for img_num in tqdm(range(len(os.listdir(args.dataset2)) - 1)):
        img_num_new = img_num + len(os.listdir(args.dataset1)) - 1 - failed_images
        img = cv2.imread(os.path.join(args.dataset2, 'street_view_' + str(img_num) + '.jpg'))
        cropped_image = img[0:200, 0:200]
        if not all((i == (223, 227, 228)).all() for i in cropped_image):
            cv2.imwrite(os.path.join(args.new_dataset, ('street_view_' + str(img_num_new) + '.jpg')), img)
            csv_writer.writerow(data2[img_num])
        else:
            failed_images += 1
            

if __name__ == '__main__':
    main()
