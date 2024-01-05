
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
    def __init__(self, dir ,img_files, aug_transform=None, stride=256):
        self.dir = dir
        self.img_files = img_files
        self.transform = aug_transform
        self.stride = stride
        h,w = 512,512  # img shape
        self.crop_size = 256
        self.patch_per_line = (w-self.crop_size )//stride+1
        self.patch_per_colum = (h-self.crop_size )//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum
        print("patch_per_img", self.patch_per_img)

    def __len__(self):
        return len(self.img_files) #* self.patch_per_img

    def __getitem__(self, idx):

        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line

        img_path = self.dir + "/images_resized/" +  self.img_files[img_idx]
        mask_hs_path  = self.dir + "masks_png/" + self.img_files[img_idx]

        bgr = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        mask = np.expand_dims(np.array(Image.open(mask_hs_path)),0)


        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        mask2 = mask[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size].astype(np.int64)
  
        while np.all(mask2 == 255):
            patch_idx = np.random.randint(0, self.patch_per_img)
            h_idx, w_idx = patch_idx // self.patch_per_line, patch_idx % self.patch_per_line
            mask2 = mask[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size].astype(np.int64)

        return bgr, torch.from_numpy(mask2)

    