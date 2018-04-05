from __future__ import print_function
from models import LipRead
import preprocess
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim
import time

#load video into a tensor
filename = 'AFTERNOON.mp4'
torch.backends.cudnn.benchmark = True
print(torch.backends.cudnn.version())

vidframes = preprocess.load_video(filename)
temporalvolume = preprocess.bbc(vidframes)

labels = Variable(torch.LongTensor([42, 42, 42, 42, 42, 42, 42, 42, 42, 42]).cuda())
input = Variable(temporalvolume.cuda())

model = LipRead().cuda()

criterion = nn.NLLLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

startTime = time.time()

def output_iteration(i, prediction, time):
    print("Iteration: {} \n Prediction: {} \n Time: {} \n".format(i, prediction, time))

print("Starting training at {}...".format(startTime))
for i in range(0,101):
    optimizer.zero_grad()
    outputs = model(input)
    _, predicted = torch.max(outputs.data, 1)

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    if(i % 100 == 0):
        currentTime = time.time()
        output_iteration(i, predicted[0], currentTime - startTime)


    del outputs
