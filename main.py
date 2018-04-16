from __future__ import print_function
from models import LipRead
import torch
import toml
from training import Trainer
from validation import Validator

print("Loading options...")
with open('options.toml', 'r') as optionsFile:
    options = toml.loads(optionsFile.read())

if(options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
    print("Running cudnn benchmark...")
    torch.backends.cudnn.benchmark = True

#Create the model.
model = LipRead(options)

if(options["general"]["loadpretrainedmodel"]):
    model.load_state_dict(torch.load(options["general"]["pretrainedmodelpath"]))

#Move the model to the GPU.
if(options["general"]["usecudnn"]):
    model = model.cuda(options["general"]["gpuid"])

trainer = Trainer(options)
validator = Validator(options)

for epoch in range(options["training"]["startepoch"], options["training"]["epochs"]):

    if(options["training"]["train"]):
        trainer.epoch(model, epoch)

    if(options["validation"]["validate"]):
        validator.epoch(model)
