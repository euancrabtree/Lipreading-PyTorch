from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data import LipreadingDataset
from torch.utils.data import DataLoader
import os

class Validator():
    def __init__(self, options):

        self.validationdataset = LipreadingDataset("/udisk/pszts-ssd/AV-ASR-data/BBC_Oxford/lipread_mp4",
                                    "val", False)
        self.validationdataloader = DataLoader(
                                    self.validationdataset,
                                    batch_size=options["input"]["batchsize"],
                                    shuffle=options["input"]["shuffle"],
                                    num_workers=options["input"]["numworkers"],
                                    drop_last=True
                                )
        self.usecudnn = options["general"]["usecudnn"]

        self.batchsize = options["input"]["batchsize"]

        self.statsfrequency = options["training"]["statsfrequency"]

        self.gpuid = options["general"]["gpuid"]

    def epoch(self, model):
        print("Starting validation...")
        count = 0
        validator_function = model.validator_function()

        for i_batch, sample_batched in enumerate(self.validationdataloader):
            input = Variable(sample_batched['temporalvolume'])
            labels = sample_batched['label']

            if(self.usecudnn):
                input = input.cuda(self.gpuid)
                labels = labels.cuda(self.gpuid)

            outputs = model(input)

            count += validator_function(outputs, labels)

            print(count)


        accuracy = count / len(self.validationdataset)
        with open("accuracy.txt", "a") as outputfile:
            outputfile.write("\ncorrect count: {}, total count: {} accuracy: {}" .format(count, len(self.validationdataset), accuracy ))
