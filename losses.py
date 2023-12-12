import torch
import torch.nn as nn
import numpy as np
import glob

category2code = {"asphalt": 0, "ceramic": 1, "concrete": 2, "fabric": 3, "foliage": 4,
                            "food": 5, "glass": 6, "metal": 7, "paper": 8, "plaster": 9, "plastic": 10,
                            "rubber": 11, "soil": 12, "stone": 13, "water": 14, "wood": 15}

class SADPixelwise(nn.Module):
    def __init__(self):
        super(SADPixelwise, self).__init__()
        materials = glob.glob('materials/*.npy')
        materials = [torch.from_numpy(np.load(m)).float() for m in materials]
        self.code2material = {v: k for k, v in zip(materials, category2code.values())}

    def forward(self, input, target):
        """
        Spectral Angle Distance Objective modified for hyperspectral images.
        Operates on each pixel and then sums the SAD across all pixels.
        
        Params:
            input -> Output of the network (batch_size, height, width, 31)
            target -> Hyperspectral image (batch_size, height, width, 1)
            
        Returns:
            total_sad: Sum of SAD across all pixels
        """
        batch_size, bands, height, width = input.shape
        total_sad = 0.0

        for b in range(batch_size):
            for h in range(height):
                for w in range(width):
                    input_pixel = input[b, :, h, w].view(1, -1)
                    target_pixel = target[b, 0, h, w]
                    if target_pixel == 255:
                        continue
                    target_pixel = self.code2material[target_pixel.item()].view(1, -1)

                    input_norm = torch.sqrt(torch.mm(input_pixel, input_pixel.t()))
                    target_norm = torch.sqrt(torch.mm(target_pixel, target_pixel.t()))

                    summation = torch.mm(input_pixel, target_pixel.t())
                    angle = torch.acos(summation / (input_norm * target_norm))
                    total_sad += angle

        return total_sad

if __name__ == "__main__":
    sad = SADPixelwise()
    input = torch.randn(1, 3, 3, 3)
    target = torch.randn(1, 3, 3, 1)
    loss = sad(input, target)
    print(loss)
    print(sad.code2material)