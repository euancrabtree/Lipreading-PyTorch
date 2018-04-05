from __future__ import print_function
from models import LipRead
import preprocess
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim

#load video into a tensor
filename = 'AFTERNOON.mp4'
torch.backends.cudnn.benchmark = True
print(torch.backends.cudnn.version())

vidframes = preprocess.load_video(filename)
temporalvolume = preprocess.bbc(vidframes)

label = Variable(torch.LongTensor([42]).cuda())
input = Variable(temporalvolume.cuda())

model = LipRead().cuda()

criterion = nn.NLLLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


for i in range(0,1000):
    optimizer.zero_grad()
    outputs = model(input)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)

    loss = criterion(outputs, Variable(label))
    loss.backward()
    optimizer.step()

    del outputs
