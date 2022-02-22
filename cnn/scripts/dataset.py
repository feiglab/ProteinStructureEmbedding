#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Dataset(torch.utils.data.Dataset):
    def __init__(self, fn):

        data = np.genfromtxt(fn, delimiter=',')
        #print(data)
        self.label = data[:,0].astype(np.float32) # rgyr (could be tdiff,rdiff,vol)
        self.input1 = data[:,1:-1].reshape((-1, 1, 1000))/10.0

    def __len__(self):
        return self.label.shape[0]
    def __getitem__(self, index):
        input = self.input1[index].astype(np.float32)
        label = self.label[index]
        return input, label

