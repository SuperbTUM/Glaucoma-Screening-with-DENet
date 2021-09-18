import torch
from UNet import ResidualBlockSE, SEBlock
from dataset import Resize2_640
import torch.nn as nn


def multi_inputs(images):
    assert images.shape[0] == 3
    # (3, 128, 128, 3) as shape of raw inputs, tensor / ndarray
    resized_images = Resize2_640(size=(128, 128))(images)
    return resized_images


class Inputs(nn.Module):
    def __init__(self):
        super(Inputs, self).__init__()
        self.pre_conv11 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.pre_activate11 = nn.ReLU(inplace=True)
        self.pre_conv12 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.pre_activate12 = nn.ReLU(inplace=True)

        self.pre_conv21 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.pre_activate21 = nn.ReLU(inplace=True)
        self.pre_conv22 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pre_activate22 = nn.ReLU(inplace=True)

        self.pre_conv31 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.pre_activate31 = nn.ReLU(inplace=True)
        self.pre_conv32 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pre_activate32 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)

    def forward(self, x):
        img1, img2, img3 = x
        img1 = self.pre_conv11(img1)
        img1 = self.pre_activate11(img1)
        img1 = self.pre_conv12(img1)
        img1 = self.pre_activate12(img1)

        img2 = self.pre_conv21(img2)
        img2 = self.pre_activate21(img2)
        img2 = self.pre_conv22(img2)
        img2 = self.pre_activate22(img2)

        img3 = self.pre_conv31(img3)
        img3 = self.pre_activate31(img3)
        img3 = self.pre_conv32(img3)
        img3 = self.pre_activate32(img3)

        inputs = torch.cat((img1, img2, img3), dim=1)

        return inputs


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activate2 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)

        self.seblock = SEBlock(in_channels=out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.activate3 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = self.deconv(x)
        branch = x
        x = self.bn2(x)
        x = self.activate2(x)
        x = self.conv2(x)
        scale = self.seblock(x)
        x = x * scale + branch
        x = self.bn3(x)
        x = self.activate3(x)
        return x


class XUNet(nn.Module):
    def __init__(self):
        super(XUNet, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.residual_block1 = ResidualBlockSE(in_channels=64, out_channels=64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.residual_block2 = ResidualBlockSE(in_channels=64, out_channels=128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.residual_block3 = ResidualBlockSE(in_channels=128, out_channels=256)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.residual_block4 = ResidualBlockSE(in_channels=256, out_channels=512)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)

        self.decoder1 = Decoder(in_channels=512, out_channels=512)
        self.decoder2 = Decoder(in_channels=512, out_channels=256)
        self.decoder3 = Decoder(in_channels=256, out_channels=128)
        self.decoder4 = Decoder(in_channels=128, out_channels=64)
        self.decoder5 = Decoder(in_channels=64, out_channels=32)

        self.after_conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.activate1 = nn.ReLU(inplace=True)
        self.after_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.activate2 = nn.ReLU(inplace=True)
        self.after_conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        branch = x
        x = self.maxpool1(x)
        res1 = self.residual_block1(x)
        x = self.maxpool2(res1)
        res2 = self.residual_block2(x)
        x = self.maxpool3(res2)
        res3 = self.residual_block3(x)
        x = self.maxpool4(res3)
        res4 = self.residual_block4(x)
        x = self.maxpool5(x)

        dec1 = self.decoder1(torch.cat((x, res4), dim=1))
        dec2 = self.decoder2(torch.cat((dec1, res3), dim=1))
        dec3 = self.decoder3(torch.cat((dec2, res2), dim=1))
        dec4 = self.decoder4(torch.cat((dec3, res1), dim=1))
        dec5 = self.decoder5(torch.cat((dec4, branch), dim=1))

        res = self.after_conv1(dec5)
        res = self.activate1(res)
        res = self.after_conv2(res)
        res = self.activate2(res)
        res = self.after_conv3(res)

        return res
