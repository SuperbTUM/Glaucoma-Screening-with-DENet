import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary


class ConvPac1_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvPac1_1, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, OS=16):
        super(Encoder, self).__init__()
        self.OS = OS
        if self.OS == 16:
            strides = (6, 12, 18)
        elif self.OS == 8:
            strides = (12, 24, 36)
        else:
            raise NotImplementedError
        self.convpac1 = ConvPac1_1(in_channels=in_channels, out_channels=256)
        self.sepconv2 = SepConvBN(in_channels=in_channels, out_channels=256, dilation=strides[0], padding=strides[0])
        self.sepconv3 = SepConvBN(in_channels=in_channels, out_channels=256, dilation=strides[1], padding=strides[1])
        self.sepconv4 = SepConvBN(in_channels=in_channels, out_channels=256, dilation=strides[2], padding=strides[2])

        self.convpac5 = ConvPac1_1(in_channels=in_channels, out_channels=256)
        self.convpac6 = ConvPac1_1(in_channels=256*4, out_channels=256)
        self.last = ConvPac1_1(in_channels=256, out_channels=out_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        _, _, H, W = x.shape
        feature1 = self.convpac1(x)

        feature2 = self.sepconv2(x)

        feature3 = self.sepconv3(x)

        feature4 = self.sepconv4(x)

        feature5 = F.adaptive_avg_pool2d(x, output_size=(H // self.OS, W // self.OS))
        feature5 = self.convpac5(feature5)
        upsampled = F.upsample(feature5, size=(H, W), mode='bilinear')

        output_ = torch.cat([feature1, feature2, feature3, feature4, upsampled], dim=1)
        output = self.convpac6(output_)
        output = self.last(output)

        output = self.dropout(output)

        return output  # channel = 256 in this case


class SepConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, padding=1):
        super(SepConvBN, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                    kernel_size=kernel_size, padding=padding, dilation=dilation,
                                    groups=in_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.element_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.relu(x)
        x = self.element_conv(x)
        x = self.bn(x)
        x = self.relu2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, backbone='resnet50'):
        super(Decoder, self).__init__()
        if backbone == "resnet50":
            self.conv1 = nn.Conv2d(in_channels=512, out_channels=48, kernel_size=1)
        elif backbone == "xception":
            self.conv1 = nn.Conv2d(in_channels=256, out_channels=48, kernel_size=1)
        else:
            raise NotImplementedError
        self.bn1 = nn.BatchNorm2d(48)
        self.relu1 = nn.ReLU(inplace=True)
        self.sepconv1 = SepConvBN(in_channels=in_channels, out_channels=256)
        self.sepconv2 = SepConvBN(in_channels=256, out_channels=256)

        self.avepool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, out_channels)

        self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)

    def forward(self, x_packed):
        conv_stride_4, x = x_packed
        conv_stride_4 = self.conv1(conv_stride_4)
        conv_stride_4 = self.bn1(conv_stride_4)
        conv_stride_4 = self.relu1(conv_stride_4)
        _, _, H, W = x.shape
        x = F.upsample(x, size=(H // 4, W // 4), mode="bilinear")
        dec = torch.cat([x, conv_stride_4], dim=1)
        dec = self.sepconv1(dec)
        dec = self.sepconv2(dec)

        dec = self.avepool(dec)
        dec = self.fc(dec)
        dec = self.softmax(dec)
        return dec


# assuming backbone: ResNet, OS=16
class DeepLabV3(nn.Module):
    def __init__(self, in_channels, intermidiate_channels, out_channels, OS=16):
        super(DeepLabV3, self).__init__()
        model = list(models.resnet50(pretrained=True).children())
        self.resnet_stride_4 = nn.Sequential(*model[:6])
        self.resnet_stride_16 = nn.Sequential(*model[6:-2])
        self.encoder = Encoder(in_channels=in_channels, out_channels=intermidiate_channels, OS=OS)
        self.decoder = Decoder(in_channels=intermidiate_channels, out_channels=out_channels)

    def forward(self, x):
        feature_stride_4 = self.resnet_stride_4(x)
        feature_stride_16 = self.resnet_stride_16(feature_stride_4)
        output = self.encoder(feature_stride_16)
        inputs = (feature_stride_4, output)
        result = self.decoder(inputs)
        return result


if __name__ == "__main__":
    model = DeepLabV3(in_channels=3, intermidiate_channels=256, out_channels=2)
    print(summary(model))
