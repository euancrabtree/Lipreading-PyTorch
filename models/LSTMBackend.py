import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMBackend(nn.Module):
    def __init__(self):
        super(LSTMBackend, self).__init__()
        self.Module1 = nn.LSTM(input_size=256, hidden_size=256, num_layers=2,batch_first=True,  bidirectional=True)
        self.fc = nn.Linear(512, 500)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):

        temporalDim = 1

        lstmOutput, _ = self.Module1(input)

        output = self.fc(lstmOutput[:, -1, :])
        output = self.softmax(output)

        return output
