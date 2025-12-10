import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


class PreprocessingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, hr_batch):
        tmp = gaussian_blur(hr_batch, kernel_size=[5, 5])
        tmp = F.interpolate(tmp, scale_factor=0.25, mode="bicubic")
        tmp = F.interpolate(tmp, size=hr_batch.shape[-2:], mode="bicubic")
        return tmp
