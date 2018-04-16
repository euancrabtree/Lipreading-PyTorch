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
import training

gpuid = 2

print("Loading options...")
with open('options.toml', 'r') as optionsFile:
    options = toml.loads(optionsFile.read())

if(options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
    print("Running cudnn benchmark...")
    torch.backends.cudnn.benchmark = True

validationdataset = LipreadingDataset("/udisk/pszts-ssd/AV-ASR-data/BBC_Oxford/lipread_mp4",
                            "val")
validationdataloader = DataLoader(validationdataset, batch_size=options["input"]["batchsize"],
                        shuffle=options["input"]["shuffle"],
                        num_workers=options["input"]["numworkers"],
                        drop_last=True)

#Create the model.
model = LipRead(options)

if(options["general"]["loadpretrainedmodel"]):
    model.load_state_dict(torch.load(options["general"]["pretrainedmodelpath"]))

trainer = Trainer(options)

for epoch in range(options["training"]["startepoch"], options["training"]["epochs"]):

    if(options["training"]["train"]):
        trainer.epoch(model)

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
