from __future__ import print_function
from models import LipRead
import preprocess
import pylab
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim

#load video into a tensor
filename = 'AFTERNOON.mp4'
label = 42

desiredOutput = torch.zeros(500)
desiredOutput[label] = 1

vidframes = preprocess.load_video(filename)
temporalvolume = preprocess.bbc(vidframes)

model = LipRead().cuda()

#criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

optimizer.zero_grad()
outputs = model(temporalvolume)

loss = criterion(outputs, Variable(torch.LongTensor([42])))
loss.backward()
optimizer.step()


for i in range(0,100000):
    optimizer.zero_grad()
    outputs = model(temporalvolume.cuda())

    loss = criterion(outputs, Variable(torch.LongTensor([42]).cuda()))
    loss.backward()
    optimizer.step()

output = model(temporalvolume)
print(output)
