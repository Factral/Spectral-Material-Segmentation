
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import transforms

class LocalMatDataset(Dataset):
    def __init__(self, dir ,img_files, aug_transform=None, stride=8):
        self.dir = dir
        self.img_files = img_files
        self.transform = aug_transform
        self.stride = stride
        h,w = 512,482  # img shape
        self.crop_size = 128
        self.patch_per_line = (w-self.crop_size )//stride+1
        self.patch_per_colum = (h-self.crop_size )//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum

    def __len__(self):
        return len(self.img_files) * self.patch_per_img

    def __getitem__(self, idx):

        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line

        img_path = self.dir + "/images_resized/" +  self.img_files[img_idx]
        mask_path  = self.dir + "/materials_hs/" + self.img_files[img_idx].replace(".png", ".pt")


        bgr = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        hyper = torch.load(mask_path).float()

        if self.transform:
            transformed = self.transform(image=bgr, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        #bgr = self.bgrs[img_idx]
        #hyper = self.hypers[img_idx]
        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]

        bgr = bgr + (bgr ==0) * 1e-4
        hyper = hyper + (hyper ==0) * 1e-4

        bgr = bgr - (bgr ==1) * 1e-4
        hyper = hyper - (hyper ==1) * 1e-4

        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    