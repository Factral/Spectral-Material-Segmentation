
import torch
import torch.nn as nn

import numpy as np
import os
import glob
import spectral

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / label
        mrae = torch.mean(error.view(-1))
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.view(-1)))
        return rmse

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)


category2code = {"asphalt": 0, "ceramic": 1, "concrete": 2, "fabric": 3, "foliage": 4, "food": 5, "glass": 6, "metal": 7, "paper": 8, "plaster": 9, "plastic": 10,"rubber": 11, "soil": 12, "stone": 13, "water": 14, "wood": 15}

class HsiMaterial():
    def __init__(self):

        category2code = {"asphalt": 0, "ceramic": 1, "concrete": 2, "fabric": 3, "foliage": 4,
                            "food": 5, "glass": 6, "metal": 7, "paper": 8, "plaster": 9, "plastic": 10,
                            "rubber": 11, "soil": 12, "stone": 13, "water": 14, "wood": 15}
        
        materials = glob.glob('materials_numpy/*.npy')
        materials = sorted(materials)

        self.materials = np.array([np.load(m) for m in materials])
        self.code2material = {v: k for k, v in zip(materials, category2code.values())}
        self.num_bands = 31

    def convert(self, cube):
        """
        Convert a hyperspectral cube to a material cube

        Params:
            cube -> Hyperspectral cube (batch_size, height, width, 31)

        Returns:
            material_cube: Material cube (batch_size, height, width, 1)
        """
        cube = cube.transpose(1,2,0)
        print(cube.shape, self.materials.shape)
        assert cube.shape[-1] == self.num_bands


        result_sam = spectral.algorithms.spectral_angles(cube, self.materials)

        return np.argmin(result_sam, axis=2)
    

def make_plot(inputs, labels, outputs, mask, hsimaterial):
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    gt = hsimaterial.convert(labels[0].cpu().numpy()) * mask[0].cpu().numpy()
    im = ax[1].imshow(gt, vmin=0, vmax=15)
    ax[1].set_title("Ground Truth")

    pred = hsimaterial.convert(outputs[0].detach().cpu().numpy())
    im = ax[2].imshow(pred , vmin=0, vmax=15)
    ax[2].set_title("Prediction")

    values = np.unique(pred)
    print(values)

    colors = [ im.cmap(im.norm(value)) for value in values]
    aa = [ mpatches.Patch(color=colors[i], label=f"{list(category2code.keys())[list(category2code.values()).index(val)]}") for i,val in enumerate(values) ]

    plt.legend(handles=aa, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.2)

    im = ax[0].imshow(inputs[0].cpu().numpy().transpose(1,2,0))
    ax[0].set_title("Input")

    return fig