import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResStack(nn.Module):
    def __init__(self, channel):
        super(ResStack, self).__init__()

        block_list = [
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(3**i),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=3, dilation=3**i)),
                nn.LeakyReLU(0.2),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1)),
            )
            for i in range(3)
        ]

        self.block0 = block_list[0]
        self.block1 = block_list[1]
        self.block2 = block_list[2]

        shortcut_list = [
            nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1))
            for i in range(3)
        ]

        self.shortcut0 = shortcut_list[0]
        self.shortcut1 = shortcut_list[1]
        self.shortcut2 = shortcut_list[2]

    def forward(self, x):
        x = self.shortcut0(x) + self.block0(x)
        x = self.shortcut1(x) + self.block1(x)
        x = self.shortcut2(x) + self.block2(x)
        return x

