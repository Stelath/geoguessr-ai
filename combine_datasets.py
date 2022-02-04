import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset1", help="The first dataset folder", required=True, type=str)
    parser.add_argument("--dataset2", help="The second dataset folder", type=str)
    return parser.parse_args()

args = get_args()

def main():
    for image in enumerate(tqdm(os.listdir(args.dataset1))):
        img = cv2.imread(os.path.join(args.dataset1, image))
        cropped_image = img[0:200, 0:200]
        if all(i == (228, 227, 223) for i in my_list1)


if __name__ == '__main__':
    main()
