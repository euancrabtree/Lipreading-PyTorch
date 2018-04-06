from __future__ import print_function
from models import LipRead
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim
import time
from data import LipreadingDataset
from torch.utils.data import DataLoader
import re
import os

torch.backends.cudnn.benchmark = True

#Create the model on the GPU.
model = LipRead().cuda()
#model = nn.DataParallel(model,device_ids=[0])

dataset = LipreadingDataset(0, 0)
dataloader = DataLoader(dataset, batch_size=10,
                        shuffle=True, num_workers=10)

criterion = nn.NLLLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

startTime = time.time()

def output_iteration(i, prediction, time):
    os.system('clear')

    print(time.__class__.__name__)
    avgBatchTime = time / (i+1)
    estTime = avgBatchTime * (len(dataset) - i)
    print("Iteration: {}\nElapsed Time: {} \nEstimated Time Remaining: {}".format(i, time, estTime))

print("Starting training...")
for i_batch, sample_batched in enumerate(dataloader):
    optimizer.zero_grad()
    input = Variable(sample_batched['temporalvolume'].cuda())
    labels = Variable(sample_batched['label'].cuda())
    outputs = model(input)


    loss = criterion(outputs, labels.squeeze(1))
    loss.backward()
    optimizer.step()

    if((i_batch * 10) % 100 == 0):
        _, predicted = torch.max(outputs.data, 1)
        currentTime = time.time()
        output_iteration(i_batch * 10, predicted[0], currentTime - startTime)
