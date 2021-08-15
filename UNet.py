import torch
import torch.nn as nn
from torchsummary import summary


class ConvPac(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Conv(x)
        x = self.relu(x)
        return x


def Encoder(in_channels, out_channels, num_encoder):
    layers = [ConvPac(in_channels, out_channels)]
    for _ in range(num_encoder-1):
        layers.append(ConvPac(out_channels, out_channels))
    layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
    layers.append(nn.MaxPool2d(kernel_size=2))  # downsample
    return nn.Sequential(*layers)


def BottomLayer(in_channels, out_channels):
    layers = [ConvPac(in_channels, out_channels),
              ConvPac(out_channels, out_channels)
              ]
    outputs = layers + [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)]
    return nn.Sequential(*outputs), nn.Sequential(*outputs[:-1])


class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.deconv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Conv1 = ConvPac(in_channels, in_channels // 2)
        self.Conv2 = ConvPac(in_channels // 2, out_channels)
        self.Conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        return x


class DecoderStage(nn.Module):
    def __init__(self, in_channels, med_channels, out_channels):
        super().__init__()
        self.decoder = Decoder(in_channels, med_channels)
        self.upsample = Upsampling(med_channels, out_channels)

    def forward(self, x):
        x = self.decoder(x)
        x = self.upsample(x)
        return x


class LocalizationNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.activate(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()
        self.encoder1 = Encoder(in_channels, out_channels=32, num_encoder=1)
        self.encoder2 = Encoder(in_channels=32, out_channels=64, num_encoder=2)
        self.encoder3 = Encoder(in_channels=64, out_channels=128, num_encoder=2)
        self.encoder4 = Encoder(in_channels=128, out_channels=256, num_encoder=2)
        self.bottom, self.branch = BottomLayer(in_channels=256, out_channels=512)
        self.upsample1 = Upsampling(in_channels=512, out_channels=256)
        self.decoder1 = DecoderStage(in_channels=512, med_channels=256, out_channels=128)
        self.decoder2 = DecoderStage(in_channels=256, med_channels=128, out_channels=64)
        self.decoder3 = DecoderStage(in_channels=128, med_channels=64, out_channels=32)
        self.decoder4 = Decoder(in_channels=64, out_channels=32)
        self.localization = LocalizationNet(in_channels=32, out_channels=1)

    def forward(self, x):
        first = self.encoder1(x)
        second = self.encoder2(first)
        third = self.encoder3(second)
        fourth = self.encoder4(third)
        bottom = self.bottom(fourth)
        bottom = self.upsample1(bottom)
        de_first = self.decoder1(torch.cat((fourth, bottom), 1))
        de_second = self.decoder2(torch.cat((third, de_first), 1))
        de_third = self.decoder3(torch.cat((second, de_second), 1))
        de_fourth = self.decoder4(torch.cat((first, de_third), 1))
        localization_results = self.localization(de_fourth)

        return localization_results


if __name__ == "__main__":
    a, b = BottomLayer(10, 10)
    print(a, b)

