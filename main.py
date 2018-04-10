from __future__ import print_function
from models import LipRead
from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data import LipreadingDataset
from torch.utils.data import DataLoader
import re
import os
import toml

print("Loading options...")
with open('options.toml', 'r') as optionsFile:
    options = toml.loads(optionsFile.read())

if(options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
    print("Running cudnn benchmark...")
    torch.backends.cudnn.benchmark = True


#load the dataset.
dataset = LipreadingDataset(0, 0)
dataloader = DataLoader(dataset, batch_size=options["input"]["batchsize"],
                        shuffle=options["input"]["shuffle"],
                        num_workers=options["input"]["numworkers"])

#Create the model.
model = LipRead(options)


#set up the loss function.
criterion = model.loss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#transfer the model to the GPU.
if(options["general"]["usecudnn"]):
    model = model.cuda()
    criterion = criterion.cuda()


startTime = datetime.now()

def timedelta_string(timedelta):
    totalSeconds = int(timedelta.total_seconds())
    hours, remainder = divmod(totalSeconds,60*60)
    minutes, seconds = divmod(remainder,60)
    return "{} hrs, {} mins, {} secs".format(hours, minutes, seconds)

def output_iteration(i, time):
    os.system('clear')

    avgBatchTime = time / (i+1)
    estTime = avgBatchTime * (len(dataset) - i)

    print("Iteration: {}\nElapsed Time: {} \nEstimated Time Remaining: {}".format(i, timedelta_string(time), timedelta_string(estTime)))

print("Starting training...")

for i in range(0, options["training"]["epochs"]):
    for i_batch, sample_batched in enumerate(dataloader):
        optimizer.zero_grad()
        input = Variable(sample_batched['temporalvolume'])
        labels = Variable(sample_batched['label'])

        if(options["general"]["usecudnn"]):
            input = input.cuda()
            labels = labels.cuda()

        outputs = model(input)
        loss = criterion(outputs, labels.squeeze(1))

        loss.backward()
        optimizer.step()
        sampleNumber = i_batch * options["input"]["batchsize"]

        if(sampleNumber % options["training"]["statsfrequency"] == 0):
            #_, predicted = torch.max(outputs.data, 2)
            currentTime = datetime.now()
            output_iteration(sampleNumber, currentTime - startTime)

    print("Starting testing...")
