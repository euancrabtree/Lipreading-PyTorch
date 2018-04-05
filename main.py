from __future__ import print_function
from models import LipRead
import preprocess
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim
import time
import re

#load video into a tensor
filename = 'AFTERNOON.mp4'
torch.backends.cudnn.benchmark = True

vidframes = preprocess.load_video(filename)
temporalvolume = preprocess.bbc(vidframes)

labels = Variable(torch.LongTensor([42, 42, 42, 42, 42, 42, 42, 42, 42, 42]).cuda())
input = Variable(temporalvolume.cuda())

#Create the model on the GPU.
model = LipRead().cuda()

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
model.apply(weights_init)

criterion = nn.NLLLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

startTime = time.time()

def output_iteration(i, prediction, time):
    print("Iteration: {} \n Prediction: {} \n Time: {} \n".format(i, prediction, time))

print("Starting training...")
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
