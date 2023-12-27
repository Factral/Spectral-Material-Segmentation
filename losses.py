import torch
import torch.nn as nn
import numpy as np
import glob



class SADPixelwise(nn.Module):
    def __init__(self,device):
        super(SADPixelwise, self).__init__()
        category2code = {"asphalt": 0, "ceramic": 1, "concrete": 2, "fabric": 3, "foliage": 4,
                                    "food": 5, "glass": 6, "metal": 7, "paper": 8, "plaster": 9, "plastic": 10,
                                    "rubber": 11, "soil": 12, "stone": 13, "water": 14, "wood": 15}

        materials = glob.glob('materials/*.npy')
        materials = [torch.from_numpy(np.load(m)).float().to(device) for m in materials]
        self.code2material = {v: k for k, v in zip(materials, category2code.values())}
        self.num_bands = 31

    def forward(self, input, target):
        """
        Spectral Angle Distance Objective modified for hyperspectral images.
        Operates on each pixel and then sums the SAD across all pixels.

        Params:
            input -> Output of the network (batch_size, height, width, 31)
            target -> Hyperspectral image (batch_size, height, width, 31)

        Returns:
            total_sad: Sum of SAD across all pixels
        """

        normalize_r = torch.norm(input, p=2, dim=3)
        normalize_g = torch.norm(target, p=2, dim=3)


        #numerator = torch.sum(torch.mul(input, target), dim=3)
        numerator =torch.einsum('ijkl,ijkl->ijk', input, target)

        if torch.sum(torch.isnan(numerator)).item() != 0 or torch.sum(torch.isnan(normalize_g)).item() != 0 or torch.sum(torch.isnan(normalize_r)).item() != 0:
            print(torch.sum(torch.isnan(input)).item(), "input")
            print(torch.sum(torch.isnan(target)).item(), "target")
            print(torch.sum(torch.isnan(numerator)).item(), "numerator")

        elemnt = numerator / ((normalize_r * normalize_g) + 1e-6)


        sad = torch.acos(elemnt)


        return torch.sum(sad)
    
class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / label
        error = error.nan_to_num()
        mrae = torch.mean(error.reshape(-1))
        return mrae


class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)) + 1e-6)
        return rmse