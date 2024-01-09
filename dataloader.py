
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

        #stride = self.stride
        #crop_size = self.crop_size
        #img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        #h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line

        img_path = self.dir + "/images_resized/" +  self.img_files[idx]
        mask_hs_path  = self.dir + "masks_png/" + self.img_files[idx]

        bgr = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        mask = np.expand_dims(np.array(Image.open(mask_hs_path)),0)


        #bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        #mask2 = mask[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size].astype(np.int64)
  
        #while np.all(mask2 == 255):
        #    patch_idx = np.random.randint(0, self.patch_per_img)
        #    h_idx, w_idx = patch_idx // self.patch_per_line, patch_idx % self.patch_per_line
        #    mask2 = mask[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size].astype(np.int64)

        return bgr, torch.from_numpy(mask)

    

class DmsDataset(Dataset):
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

        self.merge_dict = {0: {0, 14, 22, 25, 28, 31, 40, 42, 45, 54, 55}}

        self.class_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 23: 20, 24: 21, 26: 22, 27: 23, 29: 24, 30: 25, 32: 26, 33: 27, 34: 28, 35: 29, 36: 30, 37: 31, 38: 32, 39: 33, 41: 34, 43: 35, 44: 36, 46: 37, 47: 38, 48: 39, 49: 40, 50: 41, 51: 42, 52: 43, 53: 44, 56: 45}

        self.valid_classes = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23,
    24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44, 46, 47, 48, 49,
    50, 51, 52, 53, 56, ]

    def __len__(self):
        return len(self.img_files) #* self.patch_per_img

    def encode_segmap(self, mask):
        # combine categories
        for i, j in self.merge_dict.items():
            for k in j:
                mask[mask == k] = i

        # assign ignored index
        mask[mask == 0] = 255

        # map valid classes to id
        # use valid_classes sorted so will not remap.
        for valid_class in self.valid_classes:
            assert valid_class > self.class_map[valid_class]
            mask[mask == valid_class] = self.class_map[valid_class]

        return mask

    def __getitem__(self, idx):

        #stride = self.stride
        #crop_size = self.crop_size
        #img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        #h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line

        img_path = self.dir + "images_resized/" +  self.img_files[idx]

        mask_hs_path  = self.dir + "labels_resized/" + self.img_files[idx]
        mask_hs_path = mask_hs_path.replace(".jpg", ".png")

        bgr = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_hs_path))[:,:,0]
        mask = self.encode_segmap(mask)


        #bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        #mask2 = mask[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size].astype(np.int64)
  
        #while np.all(mask2 == 255):
        #    patch_idx = np.random.randint(0, self.patch_per_img)
        #    h_idx, w_idx = patch_idx // self.patch_per_line, patch_idx % self.patch_per_line
        #    mask2 = mask[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size].astype(np.int64)

        return bgr, torch.from_numpy(mask)

    