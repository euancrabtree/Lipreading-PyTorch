import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .ConvFrontend import ConvFrontend
from .ResNetBBC import ResNetBBC
from .LSTMBackend import LSTMBackend
from .ConvBackend import ConvBackend

class LipRead(nn.Module):
    def __init__(self):
        super(LipRead, self).__init__()
        self.frontend = ConvFrontend()
        self.resnet = ResNetBBC()
        self.lstm = LSTMBackend()
        self.convbackend = ConvBackend()

    def forward(self, input):
        output = self.convbackend(self.resnet(self.frontend(input)))

        return output
