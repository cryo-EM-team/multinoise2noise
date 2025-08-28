import torch
import torch.nn as nn
from torch.nn import functional as F


class Raw(nn.Module):
    # modified U-net from noise2noise paper
    def __init__(self):
        super(Raw, self).__init__()

    def forward(self, x):
        return x
