import torch
import torch.nn as nn
import torch.nn.functional as F

def _validate(modelOutput, labels):
    maxvalues, maxindices = torch.max(modelOutput.data, 1)

    count = 0

    for i in range(0, labels.squeeze(1).size(0)):

        if maxindices[i] == labels.squeeze(1)[i]:
            count += 1

    return count

class ConvBackend(nn.Module):
    def __init__(self, options):
        super(ConvBackend, self).__init__()

        bn_size = 256
        self.conv1 = nn.Conv1d(bn_size,2 * bn_size ,2, 2)
        self.norm1 = nn.BatchNorm1d(bn_size * 2)
        self.pool1 = nn.MaxPool1d(2, 2)

        self.conv2 = nn.Conv1d( 2* bn_size, 4* bn_size,2, 2)
        self.norm2 = nn.BatchNorm1d(bn_size * 4)

        self.linear = nn.Linear(4*bn_size, bn_size)
        self.norm3 = nn.BatchNorm1d(bn_size)
        self.linear2 = nn.Linear(bn_size, 500)

        self.loss = nn.CrossEntropyLoss()

        self.validator = _validate

    def forward(self, input):
        transposed = input.transpose(1, 2).contiguous()

        output = self.conv1(transposed)
        output = self.norm1(output)
        output = F.relu(output)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.norm2(output)
        output = F.relu(output)
        output = output.mean(2)
        output = self.linear(output)
        output = self.norm3(output)
        output = F.relu(output)
        output =self.linear2(output)

        return output
