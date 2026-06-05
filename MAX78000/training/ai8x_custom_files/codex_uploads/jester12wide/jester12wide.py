###################################################################################################
# 20BN-Jester 12-frame wider compact gesture network for MAX78000 deployment
###################################################################################################
"""12-frame Jester gesture-recognition network."""

from torch import nn

import ai8x


class Jester12WideNet(nn.Module):
    """2D CNN that treats 12 grayscale video frames as input channels."""

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

        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(
            64, 64, kernel_size=3, padding=1, bias=bias, **kwargs
        )
        dim_x //= 2
        dim_y //= 2

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
        x = self.conv4(x)
        x = self.mp(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def jester12wide(pretrained=False, **kwargs):
    """Construct a 12-frame wider JesterNet model."""
    assert not pretrained
    return Jester12WideNet(**kwargs)


models = [
    {
        "name": "jester12wide",
        "min_input": 1,
        "dim": 2,
    }
]
