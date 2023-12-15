
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
    def __init__(self, dir ,img_files, aug_transform=None):
        self.dir = dir
        self.img_files = img_files
        self.transform = aug_transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.dir + "/images_resized/" +  self.img_files[idx]
        mask_path  = self.dir + "/materials_hs/" + self.img_files[idx].replace(".png", ".pt")

        image = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        mask = torch.load(mask_path)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask.permute(2,0,1).float()
    
if __name__ == '__main__':
    train_files = np.load('train_files.npy')

    aug_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),  # Rota la imagen hasta 45 grados. Ajusta 'limit' según sea necesario
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0,0,0), std=(1,1,1)),
        ToTensorV2()
    ])

    dataset = LocalMatDataset("matbase", train_files, aug_transform=aug_transform)
    data_loader_train = DataLoader(dataset, batch_size=8, shuffle=True)

    for i, (image, mask) in enumerate(data_loader_train):
        print(image.shape)
        print(mask.shape)


        break