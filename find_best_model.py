import os
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
from geoguessr_dataset import GeoGuessrDataset

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch GeoGuessr AI Best Model Locator')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='batch size (default: 64), this is the total '
                         'batch size of the GPU')
parser.add_argument('--models-dir', default='models', type=str)

start_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
args = parser.parse_args()
all_loss = []

def fwd_pass(model, data, targets, loss_function, optimizer, train=False):
    data = data.cuda()
    targets = targets.cuda()

    if train:
        model.zero_grad()
    
    outputs = model(data)
    matches = [(torch.where(i >= 0.5, 1, 0) == j).all() for i, j in zip(outputs, targets)]
    acc = matches.count(True) / len(matches)
    loss = loss_function(outputs, targets)

    if train:
        loss.backward()
        optimizer.step()
    
    return acc, loss

def test(val_loader, model, loss_function):
    model.eval()
    acc = []
    loss = []
    
    for idx, sample in enumerate(tqdm(val_loader)):
        if idx < 256:
            data, target = sample
            batch_acc, batch_loss = fwd_pass(model, data, target, loss_function, None)
            acc.append(batch_acc)
            loss.append(batch_loss.cpu().detach().numpy())
    
    acc = np.mean(acc)
    loss = np.mean(loss)
    
    val_acc = np.mean(acc)
    val_loss = np.mean(loss)
    return val_acc, val_loss

def main():
    torch.device("cuda")
    valdir = os.path.join(args.data, 'val')
    val_dataset = GeoGuessrDataset(valdir)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=False, progress=True, num_classes=142)
    model = nn.Sequential(
        model,
        nn.Sigmoid()
    )
    model.cuda()
    
    loss_function = nn.BCELoss()
    
    for model_path in tqdm(os.listdir(args.models_dir)):
        model_path = os.path.join(args.models_dir, model_path)
        print("=> loading model '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("=> loaded model '{}' (epoch {})".format(model_path, checkpoint['epoch']))
        
        val_acc, val_loss = test(val_loader, model, loss_function)
        all_loss.append(val_loss)
        print("=> val_acc: {:.4f}, val_loss: {:.4f}".format(val_acc, val_loss))
    
    min_value = min(all_loss)
    min_index = all_loss.index(min_value)
    print("=> best model: {}, loss: {}".format(os.listdir(args.models_dir)[min_index], min_value))

if __name__ == '__main__':
    main()
