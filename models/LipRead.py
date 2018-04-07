import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import re

from .ConvFrontend import ConvFrontend
from .ResNetBBC import ResNetBBC
from .LSTMBackend import LSTMBackend
from .ConvBackend import ConvBackend

class LipRead(nn.Module):
    def __init__(self, options):
        super(LipRead, self).__init__()
        self.frontend = ConvFrontend()
        self.resnet = ResNetBBC(options)
        if(options["model"]["type"] == "temp-conv"):
            self.backend = ConvBackend(options)

        if(options["model"]["type"] == "LSTM" or options["model"]["type"] == "LSTM-init"):
            self.backend = LSTMBackend(options)

        #function to initialize the weights and biases of each module. Matches the
        #classname with a regular expression to determine the type of the module, then
        #initializes the weights for it.
        def weights_init(m):
            classname = m.__class__.__name__
            if re.search("Conv[123]d", classname):
                m.weight.data.normal_(0.0, 0.02)
            elif re.search("BatchNorm[123]d", classname):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)
            elif re.search("Linear", classname):
                m.bias.data.fill_(0)

        #Apply weight initialization to every module in the model.
        self.apply(weights_init)

    def forward(self, input):
        output = self.backend(self.resnet(self.frontend(input)))

        return output

    def loss(self):
        return self.backend.loss
