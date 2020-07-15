# Import torch modules
from torch import FloatTensor, cat
from torch.nn import Module, Conv3d, BatchNorm3d,Upsample, ReLU, L1Loss
import numpy as np

class CirularPad3d(Module):
    def __init__(self, pad):
        super(CirularPad3d, self).__init__()
        self.circular_pad_size = pad
    
    def forward(self, x):
        """
        :param x: shape [H, W, D]
        """
#         print(self.circular_pad_size)
        pad1, pad2, pad3 = self.circular_pad_size
        x = cat([x, x[:, :, 0:pad1, : ,:]], dim=2)
        x = cat([x, x[:, :, :, 0:pad2, :]], dim=3)
        x = cat([x, x[:, :, :, :, 0:pad3]], dim=4)
        x = cat([x[:, :, -2 * pad1:-pad1, :, :], x], dim=2)
        x = cat([x[:, :, :, -2 * pad2:-pad2, :], x], dim=3)
        x = cat([x[:, :, :, :, -2 * pad3:-pad3], x], dim=4)
        return x

    
class ConvLayer(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        pad = [int(np.floor(x / 2)) for x in kernel_size]
        self.circular_pad = CirularPad3d(pad)
        self.conv3d = Conv3d(in_channels, out_channels, kernel_size, stride)

        stop = 1

    def forward(self, x):
        out = self.circular_pad(x)
        out = self.conv3d(out)
        return out

    
class UpsampleConvLayer(Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = Upsample(scale_factor=upsample)

        pad = [int(np.floor(x / 2)) for x in kernel_size]
        self.circular_pad = CirularPad3d(pad)


        self.conv3d = Conv3d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.circular_pad(x_in)
        
        out = self.conv3d(out)
        return out
    

class ResidualBlock(Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, kernel_size):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=kernel_size, stride=1)
        self.in1 = BatchNorm3d(channels)
        self.conv2 = ConvLayer(channels, channels, kernel_size=kernel_size, stride=1)
        self.in2 = BatchNorm3d(channels)
        self.relu = ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class ModelTransformNet(Module):
    def __init__(self):
        super(ModelTransformNet, self).__init__()

        # Initial convolution layers
        self.conv1 = ConvLayer(1, 32, kernel_size=(3,9,9), stride=(1,1,1))
        self.in1 = BatchNorm3d(32)
        self.conv2 = ConvLayer(32, 64, kernel_size=(3, 3, 3), stride=(2,2,2))
        self.in2 = BatchNorm3d(64)
        self.conv3 = ConvLayer(64, 128, kernel_size=(3, 3, 3), stride=(1,2,2))
        self.in3 = BatchNorm3d(128)

        # Residual layers
        self.res1 = ResidualBlock(128, kernel_size=(3, 3, 3))
        self.res2 = ResidualBlock(128, kernel_size=(3, 3, 3))
        self.res3 = ResidualBlock(128, kernel_size=(3, 3, 3))
        self.res4 = ResidualBlock(128, kernel_size=(3, 3, 3))
        self.res5 = ResidualBlock(128, kernel_size=(3, 3, 3))

        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=(3, 3, 3), stride=1, upsample=(1,2,2))
        self.in4 = BatchNorm3d(64)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=(3, 3, 3), stride=1, upsample=(2,2,2))
        self.in5 = BatchNorm3d(32)
        self.deconv3 = ConvLayer(32, 1, kernel_size=(3, 9, 9), stride=1)

        # Non-linearities
        self.relu = ReLU()
        
    def forward(self, X):
        in_X = X
        y = self.relu(self.in1(self.conv1(in_X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y