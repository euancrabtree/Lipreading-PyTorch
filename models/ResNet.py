import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnetModel = resnet34(True)

    def forward(self, input):
        print(input.size())

        transposed = input.transpose(1, 2).contiguous()
        print(transposed.size())

        view = transposed.view(-1, 64, 28, 28)
        print(view.size())

        output = self.resnetModel(view)

        return output
