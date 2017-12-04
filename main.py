from __future__ import print_function
import models
import preprocess
import pylab
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

#load video into a tensor
filename = 'file.mp4'

vidframes = preprocess.load_video(filename)
temporalvolume = preprocess.bbc(vidframes)

frontend = models.ConvFrontend()

output = frontend(Variable(temporalvolume))
