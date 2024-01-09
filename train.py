import torch
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataloader import LocalMatDataset, DmsDataset
from architecture import *
from tqdm import tqdm
import wandb
from losses import SADPixelwise, Loss_MRAE, Loss_RMSE, FocalLoss
import numpy as np
import albumentations as A
from utils import HsiMaterial, make_plot_train, make_plot_val
from metrics import Metrics
from architecture.unet import UNetWithResnet50Encoder
import segmentation_models_pytorch.losses as losses

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--model', type=str, default='mst_plus_plus')
parser.add_argument('--weights', type=str, default="mst_plus_plus.pth")
parser.add_argument("--batch_size", type=int, default=12, help="batch size")
parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--outf", type=str, default='./exp/mst_plus_plus/', help='path log files')
parser.add_argument("--data_root", type=str, default='/media/simulaciones/hdsp/data/matbase/')
parser.add_argument("--gpu", type=str, default='0', help='path log files')
parser.add_argument("--exp_name", type=str, default='mst_plus_plus', help='path log files')

parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")

args = parser.parse_args()
wandb.login(key='fe0119224af6709c85541483adf824cec731879e')
wandb.init(project="material-segmentation", name=args.exp_name)
wandb.config.update(args)

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

train_files = np.load('split_data/train_files.npy')
val_files = np.load('split_data/val_files.npy')

dataset_train = DmsDataset(args.data_root, train_files)
data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
 pin_memory=True, drop_last=True)

dataset_test = DmsDataset(args.data_root, val_files)
data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False
, pin_memory=True)

#args.weights 
#model = model_generator(args.model).to(device)
model = UNetWithResnet50Encoder(31).to(device)

print('Parameters number is ', sum(param.numel() for param in model.parameters()))

optimizer = optim.AdamW(model.parameters(), lr=args.init_lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.1, verbose=True)

metric_train = Metrics('train')
metric_train.to(device)
metric_test = Metrics('test')
metric_test.to(device)
#cudnn.benchmark = True

criterion = nn.CrossEntropyLoss(ignore_index=255)
#criterion =  losses.FocalLoss("multiclass", ignore_index=255, gamma=3) #FocalLoss(gamma=3, ignore=255)
criterion = criterion.to(device)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLRWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5)

def train(model, data_loader, optimizer, lossfunc):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    running_loss = []
    steps = 200
    for batch_idx, (inputs, labels) in enumerate(tqdm(data_loader)):

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(inputs)
            
            loss = lossfunc(outputs, labels.long())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss.append(loss.item())

        with torch.no_grad():
            pred = nn.functional.softmax(outputs, dim=1)
            metric_train.update(pred.argmax(1), labels.long())
            model.sam.members.clamp_(0, 1)

        if batch_idx % steps == 0:
            fig = make_plot_train(inputs, outputs, labels)

            pixel_acc, macc, miou = metric_train.compute()
                    
            wandb.log({'step': batch_idx,
                        'train_loss': sum(running_loss) / len(running_loss),
                        'train_fig': fig, 
                        'pixel_acc': pixel_acc, 'macc': macc, 'miou': miou})
            
    scheduler.step()

    epoch_loss = sum(running_loss) / len(data_loader.dataset)
    metric_train.reset()

    return epoch_loss, fig


def validate(model, data_loader, lossfunc):
    model.eval()
    running_loss = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = lossfunc(outputs,  labels.squeeze(1).long()) 

            running_loss.append(loss.item())

            pred = nn.functional.softmax(outputs, dim=1)
            metric_test.update(pred.argmax(1), labels.long())

    pixel_acc, macc, miou = metric_test.compute()

    
    val_loss = sum(running_loss) / len(running_loss)
    scheduler.step(val_loss)

    fig = make_plot_val(inputs, outputs, labels)

    return val_loss, fig, pixel_acc, macc, miou

for epoch in range(args.epochs):
    print(f'Epoch {epoch + 1}\n-------------------------------')

    epoch_loss , _  = train(model, data_loader_train, optimizer, criterion)

    val_loss, fig_val, pixel_acc, macc, miou = validate(model, data_loader_test, criterion)

    wandb.log({'val_loss': val_loss,
            'val_fig': fig_val,
            'epoch': epoch,
            'pixel_acc_val': pixel_acc, 'macc_val': macc, 'miou_val': miou,
            'lr': optimizer.param_groups[0]['lr']
            })

    print(f'Epoch {epoch} train loss: {epoch_loss:.4f}, val loss: {val_loss:.4f}')


