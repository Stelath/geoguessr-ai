import argparse

import torch
import torch.utils.data

parser = argparse.ArgumentParser(description='PyTorch GeoGuessr AI Best Model Locator')
parser.add_argument('modelpath', metavar='DIR',
                    help='path to model')

args = parser.parse_args()

def main():
    checkpoint = torch.load(args.modelpath)
    torch.save(checkpoint['model_state_dict'], 'geoguessr_production_model.pt')

if __name__ == '__main__':
    main()