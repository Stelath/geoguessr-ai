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

from sam import SAM
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch GeoGuessr AI Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")

start_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
args = parser.parse_args()
writer = SummaryWriter()

def fwd_pass(model, data, targets, loss_function, optimizer, train=False):
    data = data.cuda()
    targets = targets.cuda()

    if train:
        model.zero_grad()
    
    outputs = model(data)
    matches = [(round_tensor(i) == round_tensor(j)).all() for i, j in zip(outputs, targets)]
    acc = matches.count(True) / len(matches)
    loss = loss_function(outputs, targets)

    if train:
        loss.backward()
        optimizer.step()
    
    if not train:
        close_matches = [((j + 0.005) >= i).all() and (i >= (j - 0.005)).all() for i, j in zip(outputs, targets)]
        close_acc = close_matches.count(True) / len(close_matches)
        return acc, loss, close_acc
    
    return acc, loss

def test(val_loader, model, loss_function, optimizer):
    random = np.random.randint(len(val_loader))
    
    model.eval()
    acc = []
    close_acc = []
    loss = []
    
    for idx, sample in enumerate(val_loader):
        if idx >= random and idx < random + 4:
            data, target = sample
            with torch.no_grad():
                val_acc, val_loss, val_close_acc = fwd_pass(model, data, target, loss_function, optimizer)
                acc.append(val_acc)
                close_acc.append(val_close_acc)
                loss.append(val_loss.cpu().numpy())
    
    val_acc = np.mean(acc)
    val_close_acc = np.mean(close_acc)
    val_loss = np.mean(loss)
    return val_acc, val_loss, val_close_acc

def train(train_loader, val_loader, model, loss_function, optimizer, epochs, scheduler):
    with open(f'models/{start_time}/model.log', 'a') as f:
        for epoch in range(epochs):
            scheduler(epoch)
            model.train()
            
            for idx, sample in enumerate(tqdm(train_loader)):
                data, target = sample
                acc, loss = fwd_pass(model, data, target, loss_function, optimizer, train=True)
            
            val_acc, val_loss, val_close_acc = test(val_loader, model, loss_function, optimizer)
            
            # Add accuracy and loss to tensorboard
            progress = len(train_loader) / idx
            writer.add_scalar('Loss/train', loss, epoch)
            writer.add_scalar('Accuracy/train', acc, epoch)
            writer.add_scalar('Loss/test', val_loss, epoch)
            writer.add_scalar('Accuracy/test', val_acc, epoch)
            writer.add_scalar('CloseAccuracy/test', val_close_acc, epoch)
            
            # Log Accuracy and Loss
            log = f'model-{epoch}, Accuracy: {round(float(acc), 2)}, Loss: {round(float(loss), 4)}, Val Accuracy: {round(float(val_acc), 2)}, Val Loss: {round(float(val_loss), 4)}, Val Close Accuracy: {round(float(val_close_acc))}\n'
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
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)
    loss_function = nn.L1Loss()
    
    if torch.cuda.is_available():
        print('Using GPU')
        torch.device("cuda")
        model = model.cuda()
        loss_function = loss_function.cuda()
    else:
        print('Using CPU')
        torch.device("cpu")
    
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    
    EPOCHS = args.epochs
    train(train_loader=train_loader, val_loader=val_loader, model=model, loss_function=loss_function, optimizer=optimizer, epochs=EPOCHS, scheduler=scheduler)
    
if __name__ == '__main__':
    main()