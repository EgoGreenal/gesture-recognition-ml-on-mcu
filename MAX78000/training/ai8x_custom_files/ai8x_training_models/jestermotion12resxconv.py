###################################################################################################
# 20BN-Jester 12-plane motion-channel residual gesture network with an extra low-resolution conv
###################################################################################################
"""Residual motion-channel Jester network with one extra 8x8 convolution."""

from torch import nn

import ai8x


class JesterMotion12ResXConvNet(nn.Module):
    """Motion12 residual CNN with an extra low-resolution conv stage for MAX78000 trials."""

    def __init__(self, num_classes=27, dimensions=(64, 64), num_channels=12, bias=True, **kwargs):
        super().__init__()

        dim_x, dim_y = dimensions

        self.conv1 = ai8x.FusedConv2dReLU(
            num_channels, 24, kernel_size=3, padding=1, bias=bias, **kwargs
        )
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(
            24, 48, kernel_size=3, padding=1, bias=bias, **kwargs
        )
        dim_x //= 2
        dim_y //= 2

        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(
            48, 64, kernel_size=3, padding=1, bias=bias, **kwargs
        )
        dim_x //= 2
        dim_y //= 2

        self.res1_conv1 = ai8x.FusedConv2dReLU(
            64, 64, kernel_size=3, padding=1, bias=bias, **kwargs
        )
        self.res1_conv2 = ai8x.Conv2d(
            64, 64, kernel_size=3, padding=1, bias=bias, **kwargs
        )
        self.res1_add = ai8x.Add()

        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(
            64, 64, kernel_size=3, padding=1, bias=bias, **kwargs
        )
        dim_x //= 2
        dim_y //= 2

        self.res2_conv1 = ai8x.FusedConv2dReLU(
            64, 64, kernel_size=3, padding=1, bias=bias, **kwargs
        )
        self.res2_conv2 = ai8x.Conv2d(
            64, 64, kernel_size=3, padding=1, bias=bias, **kwargs
        )
        self.res2_add = ai8x.Add()

        self.extra_conv = ai8x.FusedConv2dReLU(
            64, 64, kernel_size=3, padding=1, bias=bias, **kwargs
        )

        self.mp = ai8x.MaxPool2d(kernel_size=2, **kwargs)
        dim_x //= 2
        dim_y //= 2

        self.fc1 = ai8x.FusedLinearReLU(dim_x * dim_y * 64, 160, bias=True, **kwargs)
        self.fc2 = ai8x.Linear(160, num_classes, wide=True, bias=True, **kwargs)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward propagation."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        residual = x
        x = self.res1_conv1(x)
        x = self.res1_conv2(x)
        x = self.res1_add(x, residual)

        x = self.conv4(x)

        residual = x
        x = self.res2_conv1(x)
        x = self.res2_conv2(x)
        x = self.res2_add(x, residual)

        x = self.extra_conv(x)

        x = self.mp(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def jestermotion12resxconv(pretrained=False, **kwargs):
    """Construct a residual 12-plane JesterNet with one extra 8x8 conv stage."""
    assert not pretrained
    return JesterMotion12ResXConvNet(**kwargs)


models = [
    {
        "name": "jestermotion12resxconv",
        "min_input": 1,
        "dim": 2,
    }
]
