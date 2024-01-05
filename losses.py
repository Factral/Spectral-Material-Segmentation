import torch
import torch.nn as nn
import numpy as np
import glob
import torch.nn.functional as F

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


def one_hot(index, classes):
    """
    Converts an index to a one-hot vector.
    
    Args:
        index (torch.Tensor): Tensor containing the original index.
        classes (int): Number of classes.
    
    Returns:
        torch.Tensor: The transformed one-hot vector.
    
    """
    # index is flatten (during ignore) ##################
    
    size = index.size()[:1] + (classes,)
    view = index.size()[:1] + (1,)
    #####################################################

    # mask = torch.Tensor(size).fill_(0).to(device)
    mask = torch.Tensor(size).fill_(0)
    mask = mask.type_as(index)
    index = index.view(view)
    ones = 1.

    return mask.scatter_(1, index, ones)



class FocalLoss(nn.Module):
    """Focal loss implementation, from https://github.com/clcarwin/focal_loss_pytorch
    """
    def __init__(self, gamma=0, eps=1e-7, size_average=True, one_hot=True, ignore=-1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        
        self.size_average = size_average
        self.one_hot = one_hot
        self.ignore = ignore

    def tensor_forward(self, input, target, softmax=False):
        """
        Args:
            input (torch.Tensor): Predicted tensor with shape (B, C, H, W).
            target (torch.Tensor): Ground truth tensor with shape (B, C, H, W).
            softmax (bool, optional): Whether to perform softmax operation on the output. The default is False.
        
        Returns:
            focal loss and number of labelled samples.
        
        """
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        target = target.view(-1)
        if self.ignore is not None:
            valid = (target != self.ignore)
            input = input[valid]
            target = target[valid]

        if self.one_hot: target = one_hot(target, input.size(1))
        if softmax:
            probs = F.softmax(input, dim=1)
        else:
            probs = input
        probs = (probs * target).sum(1)
        probs = probs.clamp(self.eps, 1. - self.eps)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        count = len(batch_loss)

        return loss, count

    def forward(self, input, target, softmax=True):
        """
        Forward the focal loss, which supports the use of list that contains tensors with 
        varying shapes.
        
        Args:
            input (torch.Tensor): Predicted tensor with shape (B, C, H, W).
            target (torch.Tensor): Ground truth tensor with shape (B, C, H, W).
            softmax (bool, optional): Whether to perform softmax operation on the output. The default is False.
        
        Returns:
            focal loss and number of labelled samples.
        
        """
        is_list = isinstance(input, list)
        if not is_list:
            return self.tensor_forward(input, target, softmax)
        else:
            loss, count = [], []
            for idx in range(len(input)):
                loss_tmp, count_tmp = self.tensor_forward(input[idx], target[idx], softmax)
                loss.append(loss_tmp)
                count.append(count_tmp)
            count = torch.tensor(count)
            loss = torch.tensor(loss) * count
            loss = loss.sum() / count.sum()

            return loss, count.sum()