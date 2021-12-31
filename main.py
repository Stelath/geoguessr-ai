import os
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from utils.tensor_utils import round_tensor
from geoguessr_dataset import GeoGuessrDataset

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch GeoGuessr AI Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--checkpoint-step', default=1, type=int, metavar='N',
                    help='how often (in epochs) to save the model (default: 1)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='batch size (default: 64), this is the total '
                         'batch size of the GPU')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='learning rate for optimizer', dest='lr')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

start_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
args = parser.parse_args()
writer = SummaryWriter()

def fwd_pass(model, data, targets, loss_function, optimizer, train=False):
    data = data.cuda()
    targets = targets.cuda()

    if train:
        model.zero_grad()
    
    outputs = model(data)
    matches = [torch.where(i >= 0.5, 1, 0) == j for i, j in zip(outputs, targets)]
    acc = matches.count(True) / len(matches)
    loss = loss_function(outputs, targets)

    if train:
        loss.backward()
        optimizer.step()
    
    return acc, loss

def test(val_loader, model, loss_function, optimizer):
    random = np.random.randint(len(val_loader))
    
    model.eval()
    acc = []
    loss = []
    
    for idx, sample in enumerate(val_loader):
        if idx >= random and idx < random + 4:
            data, target = sample
            with torch.no_grad():
                val_acc, val_loss = fwd_pass(model, data, target, loss_function, optimizer)
                acc.append(val_acc)
                loss.append(val_loss.cpu().numpy())
    
    val_acc = np.mean(acc)
    val_loss = np.mean(loss)
    return val_acc, val_loss

def train(train_loader, val_loader, model, loss_function, optimizer, epochs):
    with open(f'models/{start_time}/model.log', 'a') as f:
        for epoch in range(epochs):
            model.train()
            
            for idx, sample in enumerate(tqdm(train_loader)):
                data, target = sample
                acc, loss = fwd_pass(model, data, target, loss_function, optimizer, train=True)
            
            val_acc, val_loss = test(val_loader, model, loss_function, optimizer)
            
            # Add accuracy and loss to tensorboard
            progress = len(train_loader) / idx
            writer.add_scalar('Loss/train', loss, epoch)
            writer.add_scalar('Accuracy/train', acc, epoch)
            writer.add_scalar('Loss/test', val_loss, epoch)
            writer.add_scalar('Accuracy/test', val_acc, epoch)
            
            # Log Accuracy and Loss
            log = f'model-{epoch}, Accuracy: {round(float(acc), 2)}, Loss: {round(float(loss), 4)}, Val Accuracy: {round(float(val_acc), 2)}, Val Loss: {round(float(val_loss), 4)}\n'
            print(log, end='')
            f.write(log)
            
            if epoch % args.checkpoint_step == 0:
                print('Saving model...')
                torch.save(model.state_dict(), f'models/{start_time}/model-{epoch}.pth')

def main():
    os.makedirs(f'models/{start_time}', exist_ok=True)
    
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    train_dataset = GeoGuessrDataset(traindir)
    val_dataset = GeoGuessrDataset(valdir)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=False, progress=True, num_classes=2)
    model = nn.Sequential(
        model,
        nn.Sigmoid()
    )
    
    loss_function = nn.BCELoss()
    
    if torch.cuda.is_available():
        print('Using GPU')
        torch.device("cuda")
        model = model.cuda()
        loss_function = loss_function.cuda()
    else:
        print('Using CPU')
        torch.device("cpu")
    
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    
    EPOCHS = args.epochs
    train(train_loader=train_loader, val_loader=val_loader, model=model, loss_function=loss_function, optimizer=optimizer, epochs=EPOCHS)
    
if __name__ == '__main__':
    main()