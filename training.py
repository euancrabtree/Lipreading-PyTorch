from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data import LipreadingDataset
from torch.utils.data import DataLoader
import os

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

class Trainer():
    def __init__(self, options):
        self.trainingdataset = LipreadingDataset(options["training"]["dataset"], "train")
        self.trainingdataloader = DataLoader(
                                    trainingdataset,
                                    batch_size=options["input"]["batchsize"],
                                    shuffle=options["input"]["shuffle"],
                                    num_workers=options["input"]["numworkers"],
                                    drop_last=True
                                )
        self.usecudnn = options["general"]["usecudnn"]

        self.batchhsize = options["input"]["batchsize"]

        self.statsfrequency = options["training"]["statsfrequency"]

    def epoch(self, model):
        #set up the loss function.
        criterion = model.loss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        #transfer the model to the GPU.
        if(self.usecudnn):
            model = model.cuda(gpuid)
            criterion = criterion.cuda(gpuid)

        startTime = datetime.now()
        print("Starting training...")
        for i_batch, sample_batched in enumerate(self.trainingdataloader):
            optimizer.zero_grad()
            input = Variable(sample_batched['temporalvolume'])
            labels = Variable(sample_batched['label'])

            if(self.usecudnn):
                input = input.cuda(gpuid)
                labels = labels.cuda(gpuid)

            outputs = model(input)
            loss = criterion(outputs, labels.squeeze(1))

            loss.backward()
            optimizer.step()
            sampleNumber = i_batch * self.batchsize

            if(sampleNumber % self.statsfrequency == 0):
                currentTime = datetime.now()
                output_iteration(sampleNumber, currentTime - startTime)

        print("Epoch completed, saving state...")
        torch.save(model.state_dict(), "trainedmodel.pt")
