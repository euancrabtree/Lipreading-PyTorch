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
        self.backend = ConvBackend(options)
        self.lstm = LSTMBackend(options)

        self.type = options["model"]["type"]

        def freeze(m):
            m.requires_grad=False

        if(options["model"]["type"] == "LSTM-init"):
            self.frontend.apply(freeze)
            self.resnet.apply(freeze)


        self.frontend.apply(freeze)
        self.resnet.apply(freeze)

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
        if(self.type == "temp-conv"):
            output = self.backend(self.resnet(self.frontend(input)))

        if(self.type == "LSTM" or self.type == "LSTM-init"):
            output = self.lstm(self.resnet(self.frontend(input)))

        return output

    def loss(self):
        if(self.type == "temp-conv"):
            return self.backend.loss

        if(self.type == "LSTM" or self.type == "LSTM-init"):
            return self.lstm.loss

    def validator(self):
        if(self.type == "temp-conv"):
            return self.backend.validator

        if(self.type == "LSTM" or self.type == "LSTM-init"):
            return self.lstm.validator
