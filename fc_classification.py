import torch
import torch.nn as nn
from UNet import UNet


class ClassificationNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avepool = nn.AvgPool2d(kernel_size=7)
        self.fc1 = nn.Linear(25 * in_channels, 2048)
        self.fc2 = nn.Linear(2048, 2)
        self.cls = nn.Softmax(dim=1)

    def forward(self, x):
        assert x.shape[2:] == (40, 40)
        x = self.avepool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.cls(x)
        return x


class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = UNet()
        for p in self.parameters():
            p.requires_grad = False
        self.classifier = ClassificationNet(in_channels=512)

    def forward(self, x):
        x = self.net.encoder1(x)
        x = self.net.encoder2(x)
        x = self.net.encoder3(x)
        x = self.net.encoder4(x)
        x = self.net.branch(x)
        x = self.classifier(x)
        return x
