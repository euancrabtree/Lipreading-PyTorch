import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBackend(nn.Module):
    def __init__(self):
        super(ConvBackend, self).__init__()

        bn_size = 256
        self.conv1 = nn.Conv1d(bn_size, bn_size * 5,2, 2)
        self.norm1 = nn.BatchNorm1d(bn_size * 2)
        self.pool1 = nn.MaxPool1d(2, 2)

        self.conv2 = nn.Conv1d( 2* bn_size, 4* bn_size * 5,2, 2)
        self.norm2 = nn.BatchNorm1d(bn_size * 4)

        self.conv3 = nn.Conv1d(bn_size, bn_size * 2,5, 2)
        self.norm3 = nn.BatchNorm1d(bn_size * 2)
        self.pool3 = nn.MaxPool1d(2, 2)



    def forward(self, input):
        print(input.size())

        output = self.conv1(input)
        output = self.norm1(output)
        output = F.ReLU(output)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.norm2(output)
        output = F.ReLU(output)
        output = output.mean(2)

        print(output.size())

        return output
