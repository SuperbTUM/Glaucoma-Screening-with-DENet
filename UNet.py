import torch
import torch.nn as nn
from torchsummary import summary


class ConvPac(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlockSE(nn.Module):
    def __init__(self, in_channels, out_channels, SE=True):
        super(ResidualBlockSE, self).__init__()
        self.conv1 = ConvPac(in_channels=in_channels, out_channels=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.skipconnect = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.se_block = SEBlock(out_channels) if SE else None
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        skipconnect = self.skipconnect(x)
        x = self.conv1(x)
        x = self.conv2(x)
        if self.se_block:
            scale = self.se_block(x)
            x += skipconnect * scale
        else:
            x += skipconnect
        x = self.bn(x)
        x = self.relu(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels):
        super(SEBlock, self).__init__()
        self.avepool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // 16, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // 16, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = self.avepool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def Encoder(in_channels, out_channels, num_encoder):
    layers = [ConvPac(in_channels, out_channels)]
    for _ in range(num_encoder-1):
        layers.append(ConvPac(out_channels, out_channels))
    layers.append(ResidualBlockSE(out_channels, out_channels))
    layers.append(nn.MaxPool2d(kernel_size=2))  # downsample
    return nn.Sequential(*layers)


def BottomLayer(in_channels, out_channels):
    layers = [ConvPac(in_channels, out_channels), ConvPac(out_channels, out_channels),
              nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)]
    return nn.Sequential(*layers), nn.Sequential(*layers[:-1])


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
        self.Conv3 = ResidualBlockSE(out_channels, out_channels, SE=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.bn(x)
        x = self.relu(x)
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
    def __init__(self, in_channels, factor):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels=1, kernel_size=1)
        self.classifier = nn.Sigmoid()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.classifier(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = Encoder(in_channels=3, out_channels=32, num_encoder=1)
        self.encoder2 = Encoder(in_channels=32, out_channels=64, num_encoder=2)
        self.encoder3 = Encoder(in_channels=64, out_channels=128, num_encoder=2)
        self.encoder4 = Encoder(in_channels=128, out_channels=256, num_encoder=2)
        self.bottom, self.branch = BottomLayer(in_channels=256, out_channels=512)
        self.upsample1 = Upsampling(in_channels=512, out_channels=256)
        self.decoder1 = DecoderStage(in_channels=512, med_channels=256, out_channels=128)
        self.decoder2 = DecoderStage(in_channels=256, med_channels=128, out_channels=64)
        self.decoder3 = DecoderStage(in_channels=128, med_channels=64, out_channels=32)
        self.decoder4 = Decoder(in_channels=64, out_channels=32)
        self.localization1 = LocalizationNet(in_channels=32, factor=1)
        self.localization2 = LocalizationNet(in_channels=32, factor=2)
        self.localization3 = LocalizationNet(in_channels=32, factor=4)
        self.localization4 = LocalizationNet(in_channels=32, factor=8)

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
        localization_results = (self.localization1(de_fourth) +
                                self.localization2(de_third) +
                                self.localization3(de_second) +
                                self.localization4(de_first)) / 4

        return localization_results


if __name__ == "__main__":
    model = UNet()
    print(summary(model))

