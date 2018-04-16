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

gpuid = 2

print("Loading options...")
with open('options.toml', 'r') as optionsFile:
    options = toml.loads(optionsFile.read())

if(options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
    print("Running cudnn benchmark...")
    torch.backends.cudnn.benchmark = True


#load the dataset.
trainingdataset = LipreadingDataset("/udisk/pszts-ssd/AV-ASR-data/BBC_Oxford/lipread_mp4",
                            "train")
trainingdataloader = DataLoader(trainingdataset, batch_size=options["input"]["batchsize"],
                        shuffle=options["input"]["shuffle"],
                        num_workers=options["input"]["numworkers"],
                        drop_last=True)

validationdataset = LipreadingDataset("/udisk/pszts-ssd/AV-ASR-data/BBC_Oxford/lipread_mp4",
                            "val")
validationdataloader = DataLoader(validationdataset, batch_size=options["input"]["batchsize"],
                        shuffle=options["input"]["shuffle"],
                        num_workers=options["input"]["numworkers"],
                        drop_last=True)

#Create the model.
model = LipRead(options)

model.load_state_dict(torch.load('epoch9.pt'))

#set up the loss function.
criterion = model.loss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#transfer the model to the GPU.
if(options["general"]["usecudnn"]):
    model = model.cuda(gpuid)
    criterion = criterion.cuda(gpuid)


startTime = datetime.now()

def timedelta_string(timedelta):
    totalSeconds = int(timedelta.total_seconds())
    hours, remainder = divmod(totalSeconds,60*60)
    minutes, seconds = divmod(remainder,60)
    return "{} hrs, {} mins, {} secs".format(hours, minutes, seconds)

def output_iteration(i, time):
    os.system('clear')

    avgBatchTime = time / (i+1)
    estTime = avgBatchTime * (len(trainingdataset) - i)

    print("Iteration: {}\nElapsed Time: {} \nEstimated Time Remaining: {}".format(i, timedelta_string(time), timedelta_string(estTime)))

for epoch in range(options["training"]["startepoch"], options["training"]["epochs"]):

    if(options["training"]["train"]):
        print("Starting training...")
        for i_batch, sample_batched in enumerate(trainingdataloader):
            optimizer.zero_grad()
            input = Variable(sample_batched['temporalvolume'])
            labels = Variable(sample_batched['label'])

            if(options["general"]["usecudnn"]):
                input = input.cuda(gpuid)
                labels = labels.cuda(gpuid)

            outputs = model(input)
            loss = criterion(outputs, labels.squeeze(1))

            loss.backward()
            optimizer.step()
            sampleNumber = i_batch * options["input"]["batchsize"]

            if(sampleNumber % options["training"]["statsfrequency"] == 0):
                currentTime = datetime.now()
                output_iteration(sampleNumber, currentTime - startTime)

        print("Epoch completed, saving state...")
        torch.save(model.state_dict(), "trainedmodel.pt")

    print("Starting validation...")
    count = 0
    validator = model.validator()
    for i_batch, sample_batched in enumerate(validationdataloader):
        optimizer.zero_grad()
        input = Variable(sample_batched['temporalvolume'])
        labels = sample_batched['label']

        if(options["general"]["usecudnn"]):
            input = input.cuda(gpuid)
            labels = labels.cuda(gpuid)

        outputs = model(input)

        count += validator(outputs, labels)

        sampleNumber = i_batch * options["input"]["batchsize"]

        if(sampleNumber % options["training"]["statsfrequency"] == 0):
            print(count)

    accuracy = count / len(validationdataset)
    with open("accuracy.txt", "a") as outputfile:
        outputfile.write("\nEpoch: {}, correct count: {}, total count: {} accuracy: {}" .format(epoch, count, len(validationdataset), accuracy ))
