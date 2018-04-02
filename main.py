from __future__ import print_function
from models import ConvFrontend
import preprocess
import pylab
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

#load video into a tensor
filename = 'AFTERNOON.mp4'

vidframes = preprocess.load_video(filename)
temporalvolume = preprocess.bbc(vidframes)

frontend = ConvFrontend()

output = frontend(Variable(temporalvolume))
