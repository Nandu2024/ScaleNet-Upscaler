import torch
import torch.nn as nn
from torchvision import models, transforms

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features[:36].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.resize = resize
        self.criterion = nn.MSELoss()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, sr, hr):
        sr = sr.clamp(0, 1)
        hr = hr.clamp(0, 1)
        if self.resize:
            sr = nn.functional.interpolate(sr, size=(224, 224), mode='bilinear', align_corners=False)
            hr = nn.functional.interpolate(hr, size=(224, 224), mode='bilinear', align_corners=False)
        sr = self.normalize(sr)
        hr = self.normalize(hr)
        return self.criterion(self.vgg(sr), self.vgg(hr))
