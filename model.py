import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerModel, SegformerConfig
from DSConv_free import DSConv_pro

class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        # x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.fc2 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.conv = nn.Conv1d(2, 1, 1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc1(self.avg_pool(x))
        max_out = self.fc2(self.max_pool(x))
        # out = avg_out + max_out
        avg_out = avg_out.squeeze(-1).reshape(avg_out.shape[0], 1, avg_out.shape[1])
        max_out = max_out.squeeze(-1).reshape(max_out.shape[0], 1, max_out.shape[1])
        out = self.conv(torch.cat([avg_out, max_out], dim=1))
        out = out.reshape(out.shape[0], out.shape[2], 1, 1)
        return self.sigmoid(out)


# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
#                                 nn.ReLU(),
#                                 nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x)
#                           )
#         out = avg_out + max_out
#         return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        # ablation
        return x


class FeatureFusion(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeatureFusion, self).__init__()
        self.cbam = CBAM(in_channel)
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x_high: torch.tensor, x_low: torch.tensor):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x_low, x_high), dim=1)
        residual = self.shortcut(x)
        out = self.cbam(x)
        # out = x  # ablation
        out = self.left(out)
        out += residual

        return out.relu_()


class DSC_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DSC_block, self).__init__()
        self.conv_0 = EncoderConv(in_channel, out_channel)
        self.conv_x = DSConv_pro(in_channel, out_channel, offset_mode="pyramid")
        self.conv_y = DSConv_pro(in_channel, out_channel, offset_mode="pyramid")
        # self.conv_x = old_DSConv_pro(in_channel, out_channel, morph=0)
        # self.conv_y = old_DSConv_pro(in_channel, out_channel, morph=1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv_cbam = CBAM(3 * out_channel)
        # self.conv_cbam = nn.Identity()  # ablation
        self.conv_out = EncoderConv(3 * out_channel, out_channel)

    def forward(self, x):
        residual = self.shortcut(x)
        f_0 = self.conv_0(x)
        f_x = self.conv_x(x)
        f_y = self.conv_y(x)
        x = self.conv_cbam(torch.cat([f_0, f_x, f_y], dim=1).relu_())
        x = self.conv_out(x)
        x += residual

        return x.relu_()


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x_low: torch.tensor):
        x = F.interpolate(x_low, scale_factor=2, mode='bilinear', align_corners=True)
        residual = self.shortcut(x)
        out = self.left(x)
        out += residual

        return out.relu_()


class allcnn_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(allcnn_block, self).__init__()
        self.conv_0 = EncoderConv(in_channel, out_channel)
        self.conv_x = EncoderConv(in_channel, out_channel)
        self.conv_y = EncoderConv(in_channel, out_channel)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv_cbam = CBAM(3 * out_channel)
        # self.conv_cbam = nn.Identity()  # ablation
        self.conv_out = EncoderConv(3 * out_channel, out_channel)

    def forward(self, x):
        residual = self.shortcut(x)
        f_0 = self.conv_0(x)
        f_x = self.conv_x(x)
        f_y = self.conv_y(x)
        x = self.conv_cbam(torch.cat([f_0, f_x, f_y], dim=1).relu_())
        x = self.conv_out(x)
        x += residual

        return x.relu_()

# DSCformer
class DSCSegFormer_pretrained(nn.Module):
    def __init__(self, n_channels=3, num_classes=2):
        super(DSCSegFormer_pretrained, self).__init__()
        self.number = 16
        # self.seg_former = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.seg_former = SegformerForSemanticSegmentation.from_pretrained("./", ignore_mismatched_sizes=True)

        # DSC init
        self.dsc_block1 = DSC_block(n_channels, self.number)  # 16
        self.dsc_block2 = DSC_block(self.number, 2 * self.number)  # 32
        self.dsc_block3 = DSC_block(2 * self.number, 4 * self.number)  # 64
        self.dsc_block4 = DSC_block(4 * self.number, 8 * self.number)  # 128
        self.dsc_block5 = DSC_block(8 * self.number, 16 * self.number)  # 256

        self.maxpooling = nn.MaxPool2d(2)

        self.fusion2_1 = FeatureFusion(1 * self.number + 112, 64)
        self.fusion4_2 = FeatureFusion(2 * self.number + 192, 112)
        self.fusion8_4 = FeatureFusion(4 * self.number + 32 + 256, 192)
        self.fusion16_8 = FeatureFusion(8 * self.number + 64 + 336, 256)
        self.fusion32_16 = FeatureFusion(16 * self.number + 256 + 160, 336)

        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def segformer_encoder(self, x):
        outputs = self.seg_former(x, output_hidden_states=True)
        hs = outputs.hidden_states
        return hs[0].relu(), hs[1].relu(), hs[2].relu(), hs[3].relu()

    def dsc_encoder(self, x):
        x_0 = self.dsc_block1(x)

        x = self.maxpooling(x_0)
        x_1 = self.dsc_block2(x)

        x = self.maxpooling(x_1)
        x_2 = self.dsc_block3(x)

        x = self.maxpooling(x_2)
        x_3 = self.dsc_block4(x)

        x = self.maxpooling(x_3)
        x_4 = self.dsc_block5(x)

        return x_0, x_1, x_2, x_3, x_4

    def forward(self, x):
        f4_sf, f8_sf, f16_sf, f32_sf = self.segformer_encoder(x[:, :3])

        f1_dsc, f2_dsc, f4_dsc, f8_dsc, f16_dsc = self.dsc_encoder(x)

        f16 = self.fusion32_16(torch.cat([f16_sf, f16_dsc], dim=1), f32_sf)

        f8 = self.fusion16_8(torch.cat([f8_sf, f8_dsc], dim=1), f16)

        f4 = self.fusion8_4(torch.cat([f4_sf, f4_dsc], dim=1), f8)

        f2 = self.fusion4_2(f2_dsc, f4)

        f1 = self.fusion2_1(f1_dsc, f2)

        out = self.out_conv(f1)
        return out.softmax(dim=1)
  


class DSCSegFormer_pretrained_allcnn(nn.Module):
    def __init__(self, n_channels=3, num_classes=2):
        super(DSCSegFormer_pretrained_allcnn, self).__init__()
        self.number = 16
        self.seg_former = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

        # DSC init
        self.dsc_block1 = allcnn_block(n_channels, self.number)  # 16
        self.dsc_block2 = allcnn_block(self.number, 2 * self.number)  # 32
        self.dsc_block3 = allcnn_block(2 * self.number, 4 * self.number)  # 64
        self.dsc_block4 = allcnn_block(4 * self.number, 8 * self.number)  # 128
        self.dsc_block5 = allcnn_block(8 * self.number, 16 * self.number)  # 256

        self.maxpooling = nn.MaxPool2d(2)

        self.fusion2_1 = FeatureFusion(1 * self.number + 112, 64)
        self.fusion4_2 = FeatureFusion(2 * self.number + 192, 112)
        self.fusion8_4 = FeatureFusion(4 * self.number + 32 + 256, 192)
        self.fusion16_8 = FeatureFusion(8 * self.number + 64 + 336, 256)
        self.fusion32_16 = FeatureFusion(16 * self.number + 256 + 160, 336)

        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def segformer_encoder(self, x):
        outputs = self.seg_former(x, output_hidden_states=True)
        hs = outputs.hidden_states
        return hs[0].relu(), hs[1].relu(), hs[2].relu(), hs[3].relu()

    def dsc_encoder(self, x):
        x_0 = self.dsc_block1(x)

        x = self.maxpooling(x_0)
        x_1 = self.dsc_block2(x)

        x = self.maxpooling(x_1)
        x_2 = self.dsc_block3(x)

        x = self.maxpooling(x_2)
        x_3 = self.dsc_block4(x)

        x = self.maxpooling(x_3)
        x_4 = self.dsc_block5(x)

        return x_0, x_1, x_2, x_3, x_4

    def forward(self, x):
        f4_sf, f8_sf, f16_sf, f32_sf = self.segformer_encoder(x[:, :3])

        f1_dsc, f2_dsc, f4_dsc, f8_dsc, f16_dsc = self.dsc_encoder(x)

        f16 = self.fusion32_16(torch.cat([f16_sf, f16_dsc], dim=1), f32_sf)

        f8 = self.fusion16_8(torch.cat([f8_sf, f8_dsc], dim=1), f16)

        f4 = self.fusion8_4(torch.cat([f4_sf, f4_dsc], dim=1), f8)

        f2 = self.fusion4_2(f2_dsc, f4)

        f1 = self.fusion2_1(f1_dsc, f2)

        out = self.out_conv(f1)
        return out.softmax(dim=1)




if __name__ == '__main__':
    a = torch.rand(1, 3, 256, 256, device="cuda", dtype=torch.float32)
    model = DSCSegFormer_pretrained()
    print(f"model: {(sum(p.numel() for p in model.parameters()))}")
    model = model.to(dtype=torch.float)
    model = model.cuda()
    aa = model(a)
    print(model)
    print(aa.shape)
