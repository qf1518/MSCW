from __future__ import absolute_import
from __future__ import division
from settings import *

import numpy as np
import random
import math
import numpy as np
import random
import PIL
import cv2
import matplotlib

# std libs
import collections
import copy
import numbers
import inspect
import shutil
from timeit import default_timer as timer
import itertools

import csv
import pandas as pd
import pickle
import glob
import sys
from distutils.dir_util import copy_tree
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime
from collections import OrderedDict


class Struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def read_list_from_file(data):
    with open(data) as file:
        lines = [line.strip() for line in file]
    return lines

PI  = np.pi
INF = np.inf
EPS = 1e-12



import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import torchvision
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel
from torch.utils import model_zoo
from torch.autograd.variable import Variable


#import keras



from training.metric import *
from training.loss import *
from training.lr_scheduler import *
from training.kaggle_metrics import *




