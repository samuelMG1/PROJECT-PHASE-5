import torch
import torch.nn as nn
import torch.nn.functional as F

# EfficientNet block with depthwise-separable convolution, batch normalization, and squeeze-and-excitation
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        """
        Initializes an MBConvBlock with depthwise separable convolution,
        batch normalization, and squeeze-and-excitation.

        Arguments:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            expansion_factor (int): Factor for expanding the input channels.
            stride (int): Stride for downsampling if needed.
        """
        super().__init__()
        mid_channels = in_channels * expansion_factor
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        # Expansion phase
        self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # Depthwise convolution
        self.dw_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # Squeeze and Excitation
        self.se_reduce = nn.Conv2d(mid_channels, mid_channels // 4, kernel_size=1)
        self.se_expand = nn.Conv2d(mid_channels // 4, mid_channels, kernel_size=1)

        # Output phase
        self.project_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.expand_conv(x)))
        out = F.relu(self.bn2(self.dw_conv(out)))

        # Squeeze-and-excitation
        se = torch.mean(out, dim=(2, 3), keepdim=True)
        se = torch.sigmoid(self.se_expand(F.relu(self.se_reduce(se))))
        out = out * se  # Scale with the SE block output

        out = self.bn3(self.project_conv(out))
        if self.use_residual:
            out = x + out
        return out


class EfficientNetMini(nn.Module):
    def __init__(self, num_classes=61, in_channels=3):
        """
        EfficientNet-inspired architecture for image classification.

        Arguments:
            num_classes (int): Number of output classes.
            in_channels (int): Number of input channels (default: 3 for RGB).
        """
        super().__init__()
        self.stem_conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(32)

        # Define a few MBConv blocks with increasing channels
        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, expansion_factor=1, stride=1),
            MBConvBlock(16, 24, expansion_factor=6, stride=2),
            MBConvBlock(24, 40, expansion_factor=6, stride=2),
            MBConvBlock(40, 80, expansion_factor=6, stride=2),
            MBConvBlock(80, 112, expansion_factor=6, stride=1),
            MBConvBlock(112, 192, expansion_factor=6, stride=2),
            MBConvBlock(192, 320, expansion_factor=6, stride=1)
        )

        # Head layers
        self.head_conv = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1280)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        out = F.relu(self.bn0(self.stem_conv(x)))
        out = self.blocks(out)
        out = F.relu(self.bn1(self.head_conv(out)))
        out = F.adaptive_avg_pool2d(out, 1).squeeze()  # Global average pooling
        out = self.classifier(out)
        return out
