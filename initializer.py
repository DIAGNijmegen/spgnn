import torch.nn as nn
import math
import torch


class Initializer:

    def initialize(self, module):
        raise NotImplementedError("need subclassing to implement.")


class HeNorm(Initializer):

    def __init__(self, **kwargs):
        self.mode = kwargs.get('mode', 'fan_in')

    def initialize(self, module):
        def init_weights(m):
            if type(m) in [nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d]:
                torch.nn.init.kaiming_normal_(m.weight, mode=self.mode)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.reset_parameters()

        module.apply(init_weights)


class XavierUniform(Initializer):

    def __init__(self, **kwargs):
        self.gain = kwargs.get('gain', nn.init.calculate_gain('relu'))

    def initialize(self, module):
        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d]:
                torch.nn.init.xavier_uniform_(m.weight, gain=self.gain)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        module.apply(init_weights)
