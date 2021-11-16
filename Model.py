import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from functools import partial

from torch.nn import init
nonlinearity = partial(F.relu, inplace=True)
up_kwargs = {'mode': 'bilinear', 'align_corners': True}



class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x
# DSC Module
class DenseSepConv(nn.Module):
    def __init__(self, in_channel=512, depth=512):
        super(DenseSepConv, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.SepConv_block1 = nn.Sequential(
            SeparableConv2d(in_channel, depth, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True))
        self.SepConv_block6 = nn.Sequential(
            SeparableConv2d(in_channel, depth, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True))
        self.SepConv_block12 = nn.Sequential(
            SeparableConv2d(in_channel, depth, kernel_size=3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True))
        self.SepConv_block18 = nn.Sequential(
            SeparableConv2d(in_channel, depth, kernel_size=3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True))

        self.conv_1x1_output_1 = nn.Sequential(
            nn.Conv2d(2 * in_channel, depth, 1, padding=0, bias=False),
            nn.BatchNorm2d(depth))
        self.conv_1x1_output_6 = nn.Sequential(
            nn.Conv2d(3 * in_channel, depth, 1, padding=0, bias=False),
            nn.BatchNorm2d(depth))
        self.conv_1x1_output_12 = nn.Sequential(
            nn.Conv2d(4 * in_channel, depth, 1, padding=0, bias=False),
            nn.BatchNorm2d(depth))
        self.conv_1x1_output_18 = nn.Conv2d(depth * 5, depth, 1, 1)
    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear', align_corners=True)
        SepConv_block1 = self.SepConv_block1(x)
        cat_block1 = self.conv_1x1_output_1(torch.cat([image_features, SepConv_block1], dim=1))

        SepConv_block6 = self.SepConv_block6(cat_block1)
        cat_block6 = self.conv_1x1_output_6(torch.cat([image_features, SepConv_block1, SepConv_block6], dim=1))

        SepConv_block12 = self.SepConv_block12(cat_block6)
        cat_block12 = self.conv_1x1_output_12(torch.cat([image_features, SepConv_block1, SepConv_block6, SepConv_block12], dim=1))

        SepConv_block18 = self.SepConv_block18(cat_block12)

        dsc = self.conv_1x1_output_18(torch.cat([image_features, SepConv_block1, SepConv_block6,
                                              SepConv_block12, SepConv_block18], dim=1))
        return dsc

# MSA Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x*self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_attn = torch.cat([avg_out, max_out], dim=1)
        x_attn = self.conv1(x_attn)
        return x*self.sigmoid(x_attn)
class MSAblock(nn.Module):
    def __init__(self, in_channels):
        super(MSAblock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, dilation=1, kernel_size=3,
                                 padding=1)

        self.bn = nn.ModuleList([nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels)])
        self.conv1x1 = nn.ModuleList(
            [nn.Conv2d(in_channels=4 * in_channels, out_channels=in_channels, dilation=1, kernel_size=1, padding=0),
             nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, dilation=1, kernel_size=1, padding=0)])
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        branches_1 = self.conv3x3(x)
        branches_1 = self.bn[0](branches_1)

        branches_2 = F.conv2d(x, self.conv3x3.weight, padding=3, dilation=3)
        branches_2 = self.bn[1](branches_2)

        branches_3 = F.conv2d(x, self.conv3x3.weight, padding=5, dilation=5)
        branches_3 = self.bn[2](branches_3)
        branches_4 = F.conv2d(x, self.conv3x3.weight, padding=7, dilation=7)
        branches_4 = self.bn[3](branches_4)
        feat = torch.cat([branches_1, branches_2, branches_3, branches_4], dim=1)
        feat = self.conv1x1[0](feat)

        feat_ca = self.ca(feat)
        feat_sa = self.sa(feat)
        feat_msa = self.conv1x1[1](torch.cat([feat_ca, feat_sa], dim=1))

        return feat_msa

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class GDCSegNet(nn.Module):
   
    def __init__(self, num_classes=3, num_channels=3):
        super(GDCSegNet, self).__init__()

        filters = [64, 64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu

        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dsc = DenseSepConv(in_channel=filters[1], depth=filters[1])

        self.decoder4 = DecoderBlock(filters[4], filters[3])
        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(filters[2], filters[1])
        self.decoder1 = DecoderBlock(filters[1], filters[1])
        self.msa = MSAblock(filters[4])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.final = nn.Sequential(
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Encoder

        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        msa_out = self.msa(e4)
        
        # Decoder
        d4 = self.decoder4(msa_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        dsc_out = self.dsc(d1)

        out = self.finaldeconv1(dsc_out)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.final(out)

        return out, out
