import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from dataloader import LocalMatDataset
from architecture import *
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt 
from metrics import Metrics
import wandb
from losses import SADPixelwise
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=2, help="batch size")
parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--outf", type=str, default='./exp/mst_plus_plus/', help='path log files')
parser.add_argument("--data_root", type=str, default='/media/simulaciones/hdsp/data/matbase/')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
parser.add_argument("--exp_name", type=str, default='mst_plus_plus', help='path log files')

args = parser.parse_args()
#wandb.login(key='b087fbd97e7a3a0e2eff2a331f13d697319a9010')
#wandb.init(project="deepbeauty1", entity="deepbeauty", name=args.exp_name)
#wandb.config.update(args)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.set_default_device(device)

aug_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),  # Rota la imagen hasta 45 grados. Ajusta 'limit' según sea necesario
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0,0,0), std=(1,1,1)),
    ToTensorV2()
])

aug_transform_test = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

train_files = np.load('train_files.npy')
test_files = np.load('test_files.npy')

dataset_train = LocalMatDataset(args.data_root, train_files)#, aug_transform=aug_transform)
data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
dataset_test = LocalMatDataset(args.data_root, test_files)#, aug_transform=aug_transform_test)
data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

model = model_generator(args.method, args.pretrained_model_path).to(device)
print('Parameters number is ', sum(param.numel() for param in model.parameters()))

optimizer = optim.Adam(model.parameters(), lr=args.init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.1, verbose=True)

metrics = Metrics()
cudnn.benchmark = True
criterion = SADPixelwise(device=device)
criterion = criterion.to(device)

if args.pretrained_model_path is not None and os.path.isfile(args.pretrained_model_path):
        print("=> loading checkpoint '{}'".format(args.pretrained_model_path))
        checkpoint = torch.load(args.pretrained_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])


def train(model, data_loader, optimizer, lossfunc):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    running_loss = 0.0
    for inputs, labels in tqdm(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(inputs)
            loss = lossfunc(outputs, labels) 

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #loss.backward()
        #optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(data_loader.dataset)

        #(acc_, ce_) = metrics.all_metrics(outputs, labels)
        #acc.append(acc_.item())
        #ce.append(ce_.item())

    return epoch_loss


def validate(model, data_loader, lossfunc):

    model.eval()
    running_loss = 0.0
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = lossfunc(outputs, labels) 

    running_loss += loss.item() * inputs.size(0)
    
        #acc_, ce_ = metrics.all_metrics(outputs, labels)
        #acc.append(acc_.item())
        #ce.append(ce_.item())

    val_loss = running_loss / len(data_loader.dataset)
    scheduler.step(val_loss)
    
    return val_loss


best_val_ce = 1000000000000000
for epoch in range(args.epochs):
    print(f'Epoch {epoch + 1}\n-------------------------------')

    epoch_loss  = train(model, data_loader_train, optimizer, criterion)
    val_loss= validate(model, data_loader_test, criterion)

    if val_loss < best_val_ce:
        best_val_ce = val_loss
        torch.save(model, os.path.join(args.save_dir, args.exp_name+'_best_model.pth'))
        torch.save(model.state_dict(), os.path.join(args.save_dir, args.exp_name+'_best_weights.pth'))
        print("Best model saved with test CE: ", best_val_ce)
    
    print(f'Epoch {epoch} train loss: {epoch_loss:.4f}, val loss: {val_loss:.4f}')
"""
    wandb.log({'epochs': epoch,
                'lr': optimizer.param_groups[0]['lr'],
                'train_loss': epoch_loss, 'val_loss': val_loss})
"""

