import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50_Mod(nn.Module):
    def __init__(self, input_size=640):
        super().__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*(list(resnet50.children())[:-2]))
        self.avepool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(int(input_size//112), 2)

    def forward(self, x):
        x = self.resnet(x)
        x = self.avepool(x)
        x = self.fc(x)
        return x
