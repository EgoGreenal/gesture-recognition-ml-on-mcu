###################################################################################################
# 20BN-Jester compact gesture network for MAX78000 deployment
###################################################################################################
"""Jester multi-frame gesture-recognition network."""

from torch import nn

import ai8x


class JesterNet(nn.Module):
    """2D CNN that treats 8 grayscale video frames as input channels."""

    def __init__(self, num_classes=27, dimensions=(64, 64), num_channels=8, bias=True, **kwargs):
        super().__init__()

        dim_x, dim_y = dimensions

        self.conv1 = ai8x.FusedConv2dReLU(
            num_channels, 16, kernel_size=3, padding=1, bias=bias, **kwargs
        )
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(
            16, 32, kernel_size=3, padding=1, bias=bias, **kwargs
        )
        dim_x //= 2
        dim_y //= 2

        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(
            32, 64, kernel_size=3, padding=1, bias=bias, **kwargs
        )
        dim_x //= 2
        dim_y //= 2

        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(
            64, 64, kernel_size=3, padding=1, bias=bias, **kwargs
        )
        dim_x //= 2
        dim_y //= 2

        self.mp = ai8x.MaxPool2d(kernel_size=2, **kwargs)
        dim_x //= 2
        dim_y //= 2

        self.fc1 = ai8x.FusedLinearReLU(dim_x * dim_y * 64, 128, bias=True, **kwargs)
        self.fc2 = ai8x.Linear(128, num_classes, wide=True, bias=True, **kwargs)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward propagation."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mp(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def jesternet(pretrained=False, **kwargs):
    """Construct a JesterNet model."""
    assert not pretrained
    return JesterNet(**kwargs)


models = [
    {
        "name": "jesternet",
        "min_input": 1,
        "dim": 2,
    }
]
