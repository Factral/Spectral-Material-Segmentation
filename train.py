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
from tqdm import tqdm
import matplotlib.pyplot as plt 
from metrics import Metrics
import wandb
from losses import SADPixelwise, Loss_MRAE
import numpy as np
import matplotlib.patches as mpatches
import matplotlib 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import HsiMaterial, make_plot
import sys

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default="mst_plus_plus.pth")
parser.add_argument("--batch_size", type=int, default=14, help="batch size")
parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--outf", type=str, default='./exp/mst_plus_plus/', help='path log files')
parser.add_argument("--data_root", type=str, default='/media/simulaciones/hdsp/data/matbase/')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
parser.add_argument("--exp_name", type=str, default='mst_plus_plus', help='path log files')
parser.add_argument("--save_dir", type=str, default='results', help='model path')


args = parser.parse_args()
wandb.login(key='fe0119224af6709c85541483adf824cec731879e')
wandb.init(project="material-segmentation", name=args.exp_name)
wandb.config.update(args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_files = np.load('train_files.npy')
test_files = np.load('test_files.npy')

dataset_train = LocalMatDataset(args.data_root, train_files)#, aug_transform=aug_transform)
data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
 pin_memory=True, drop_last=True)

dataset_test = LocalMatDataset(args.data_root, test_files)#, aug_transform=aug_transform_test)
data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False
, pin_memory=True)

model = model_generator(args.method, args.pretrained_model_path).to(device)
print('Parameters number is ', sum(param.numel() for param in model.parameters()))

optimizer = optim.Adam(model.parameters(), lr=args.init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.1, verbose=True)

metrics = Metrics()
cudnn.benchmark = True

criterion = Loss_MRAE()#SADPixelwise(device=device)
criterion = criterion.to(device)

materials = {"asphalt": 0, "ceramic": 1, "concrete": 2, "fabric": 3, "foliage": 4, "food": 5, "glass": 6, "metal": 7, "paper": 8, "plaster": 9, "plastic": 10,"rubber": 11, "soil": 12, "stone": 13, "water": 14, "wood": 15}

hsimaterial = HsiMaterial()

def train(model, data_loader, optimizer, lossfunc):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    running_loss = 0.0
    steps = 500
    for batch_idx, (inputs, labels, mask_png) in tqdm(enumerate(data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        inputs = Variable(inputs)
        labels = Variable(labels)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(inputs)

            mask = (labels != 0).any(dim=1).int()
            outputs = outputs * mask.unsqueeze(1)

            loss = lossfunc(outputs, labels) 

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)

        if batch_idx % steps == 0:
            fig = make_plot(inputs, outputs, labels, mask)
                    
            wandb.log({'step': batch_idx,
                        'train_loss': running_loss,
                        'train_fig': fig})

    return epoch_loss, fig


def validate(model, data_loader, lossfunc):

    model.eval()
    running_loss = 0.0
    #torch.cuda.empty_cache()
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = Variable(inputs)
            labels = Variable(labels)

            outputs = model(inputs)
            mask = (outputs != 0).any(dim=1).int()
            outputs = outputs * mask.unsqueeze(1)
            loss = lossfunc(outputs, labels) 

    running_loss += loss.item() * inputs.size(0)
    
        #acc_, ce_ = metrics.all_metrics(outputs, labels)
        #acc.append(acc_.item())
        #ce.append(ce_.item())

    val_loss = running_loss / len(data_loader.dataset)
    scheduler.step(val_loss)
    
    return val_loss

#best_val_ce = 1000000000000000

for epoch in range(args.epochs):
    print(f'Epoch {epoch + 1}\n-------------------------------')

    epoch_loss,fig  = train(model, data_loader_train, optimizer, criterion)
    #val_loss= validate(model, data_loader_test, criterion)


    #if val_loss < best_val_ce:
    #    best_val_ce = val_loss
    #    torch.save(model, os.path.join(args.save_dir, args.exp_name+'_best_model.pth'))
    #    torch.save(model.state_dict(), os.path.join(args.save_dir, args.exp_name+'_best_weights.pth'))
    #    print("Best model saved with test CE: ", best_val_ce)
    val_loss=0
    print(f'Epoch {epoch} train loss: {epoch_loss:.4f}, val loss: {val_loss:.4f}')

    wandb.log({'epochs': epoch,
                'lr': optimizer.param_groups[0]['lr'],
                'train_loss': epoch_loss, 'val_loss': val_loss,
                'train_fig': fig})


