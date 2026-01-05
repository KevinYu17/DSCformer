# -*- coding: utf-8 -*-
import torch
from torch import nn, cat
from torch.nn.functional import dropout
from DSConv_free import DSConv_pro

"""Dynamic Snake Convolution Network"""
# This modification was made by Kaiwei Yu
# The code is based on the ICCV 2023 Dynamic Snake Convolution

class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class DecoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)

        return x


class DSCNet_pro_free(nn.Module):
    def __init__(
            self,
            n_channels,
            n_classes,
            kernel_size,
            extend_scope,
            if_offset,
            device,
            number,
            dim,
            offset_mode,
    ):
        """
        Our DSCNet
        :param n_channels: input channel
        :param n_classes: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        :param number: basic layer numbers
        :param dim:
        """
        super().__init__()
        device = device
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.relu = nn.ReLU(inplace=True)
        self.number = number
        """
        The three contributions proposed in our paper are relatively independent. 
        In order to facilitate everyone to use them separately, 
        we first open source the network part of DSCNet. 
        <dim> is a parameter used by multiple templates, 
        which we will open source in the future ...
        """
        self.dim = dim  # This version dim is set to 1 by default, referring to a group of x-axes and y-axes
        """
        Here is our framework. Since the target also has non-tubular structure regions, 
        our designed model also incorporates the standard convolution kernel, 
        for fairness, we also add this operation to compare with other methods (like: Deformable Convolution).
        """
        self.conv00 = EncoderConv(n_channels, self.number)
        self.conv0_dsc = DSConv_pro(
            n_channels,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
            offset_mode,
        )
        self.conv1 = EncoderConv(3 * self.number, self.number)

        self.conv20 = EncoderConv(self.number, 2 * self.number)
        self.conv2_dsc = DSConv_pro(
            self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
            offset_mode,
        )
        self.conv3 = EncoderConv(6 * self.number, 2 * self.number)

        self.conv40 = EncoderConv(2 * self.number, 4 * self.number)
        self.conv4_dsc = DSConv_pro(
            2 * self.number,
            8 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
            offset_mode,
        )
        self.conv5 = EncoderConv(12 * self.number, 4 * self.number)

        self.conv60 = EncoderConv(4 * self.number, 8 * self.number)
        self.conv6_dsc = DSConv_pro(
            4 * self.number,
            16 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
            offset_mode,
        )
        self.conv7 = EncoderConv(24 * self.number, 8 * self.number)

        self.conv120 = EncoderConv(12 * self.number, 4 * self.number)
        self.conv12_dsc = DSConv_pro(
            12 * self.number,
            8 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
            offset_mode,
        )
        self.conv13 = EncoderConv(12 * self.number, 4 * self.number)

        self.conv140 = DecoderConv(6 * self.number, 2 * self.number)
        self.conv14_dsc = DSConv_pro(
            6 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv15 = DecoderConv(6 * self.number, 2 * self.number)

        self.conv160 = DecoderConv(3 * self.number, self.number)
        self.conv16_dsc = DSConv_pro(
            3 * self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
            offset_mode,
        )
        self.conv17 = DecoderConv(3 * self.number, self.number)

        self.out_conv = nn.Conv2d(self.number, n_classes, 1)
        self.maxpooling = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # block0
        x_00_0 = self.conv00(x)
        x_0_dsc = self.conv0_dsc(x)

        x_0_1 = self.conv1(torch.cat([x_00_0, x_0_dsc], dim=1))
        # x_0_1 = self.dropout(x_0_1)
        # block1
        x = self.maxpooling(x_0_1)
        x_20_0 = self.conv20(x)
        x_2_dsc = self.conv2_dsc(x)

        x_1_1 = self.conv3(torch.cat([x_20_0, x_2_dsc], dim=1))
        # x_1_1 = self.dropout(x_1_1)
        # block2
        x = self.maxpooling(x_1_1)
        x_40_0 = self.conv40(x)
        x_4_dsc = self.conv4_dsc(x)

        x_2_1 = self.conv5(torch.cat([x_40_0, x_4_dsc], dim=1))
        # x_2_1 = self.dropout(x_2_1)
        # block3
        x = self.maxpooling(x_2_1)
        x_60_0 = self.conv60(x)
        x_6_dsc = self.conv6_dsc(x)

        x_3_1 = self.conv7(torch.cat([x_60_0, x_6_dsc], dim=1))
        # x_3_1 = self.dropout(x_3_1)
        # block4
        x = self.up(x_3_1)
        x_120_2 = self.conv120(torch.cat([x, x_2_1], dim=1))
        x_12_dsc = self.conv12_dsc(torch.cat([x, x_2_1], dim=1))

        x_2_3 = self.conv13(torch.cat([x_120_2, x_12_dsc], dim=1))
        # x_2_3 = self.dropout(x_2_3)
        # block5
        x = self.up(x_2_3)
        x_140_2 = self.conv140(torch.cat([x, x_1_1], dim=1))
        x_14_dsc = self.conv14_dsc(torch.cat([x, x_1_1], dim=1))

        x_1_3 = self.conv15(torch.cat([x_140_2, x_14_dsc], dim=1))
        # x_1_3 = self.dropout(x_1_3)
        # block6
        x = self.up(x_1_3)
        x_160_2 = self.conv160(torch.cat([x, x_0_1], dim=1))
        x_16_dsc = self.conv16_dsc(torch.cat([x, x_0_1], dim=1))

        x_0_3 = self.conv17(torch.cat([x_160_2, x_16_dsc], dim=1))
        # x_0_3 = self.dropout(x_0_3)
        # out
        out = self.out_conv(x_0_3)
        out = self.softmax(out)

        return out





if __name__ == "__main__":
    model = DSCNet_pro_free(n_channels=3,
                            n_classes=3,
                            kernel_size=9,
                            extend_scope=1.0,
                            if_offset=True,
                            device="cuda",
                            number=16,
                            dim=1,
                            offset_mode="3")
    print(model)