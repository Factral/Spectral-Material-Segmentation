import torch
import torch.nn as nn
import numpy as np
import glob

category2code = {"asphalt": 0, "ceramic": 1, "concrete": 2, "fabric": 3, "foliage": 4,
                            "food": 5, "glass": 6, "metal": 7, "paper": 8, "plaster": 9, "plastic": 10,
                            "rubber": 11, "soil": 12, "stone": 13, "water": 14, "wood": 15}

class SADPixelwise(nn.Module):
    def __init__(self,device):
        super(SADPixelwise, self).__init__()
        materials = glob.glob('materials/*.npy')
        materials = [torch.from_numpy(np.load(m)).float().to(device) for m in materials]
        self.code2material = {v: k for k, v in zip(materials, category2code.values())}

    def convert_target_to_material(self, target):
        """
        Converts the target image to a material image.
        Params:
            target -> Hyperspectral image (batch_size, height, width, 1)
        Returns:
            material_image -> Material image (batch_size, height, width, wavelength)
        """
        batch_size, height, width, _ = target.shape
        target_flat = target.view(-1)
        material_flat = torch.stack([
            self.code2material[val.item()] if val.item() != 255 else torch.zeros(31)
            for val in target_flat
        ])        # Reorganizar el tensor a la forma deseada
        material_image = material_flat.view(batch_size, height, width, -1)
        return material_image

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

        # Convertir target a material
        target_material = self.convert_target_to_material(target)
        
        # Aplanar dimensiones para operaciones vectorizadas
        input_flat = input.reshape(batch_size * height * width, bands)
        target_flat = target_material.reshape(batch_size * height * width, bands)

        # Calcular normas
        input_norm = torch.norm(input_flat, dim=1, keepdim=True)
        target_norm = torch.norm(target_flat, dim=1, keepdim=True)

        # Calcular producto punto
        dot_product = torch.sum(input_flat * target_flat, dim=1, keepdim=True)

        # Calcular ángulos y sumar
        angles = torch.acos(dot_product / (input_norm * target_norm))
        total_sad = torch.sum(angles)

        return total_sad

if __name__ == "__main__":
    sad = SADPixelwise()
    input = torch.randn(1, 3, 3, 3)
    target = torch.randn(1, 3, 3, 1)
    loss = sad(input, target)
    print(loss)
    print(sad.code2material)