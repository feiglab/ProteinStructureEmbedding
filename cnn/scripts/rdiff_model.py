#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # define layers to be used
        self.conv_1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1) # convolutional layers will go over all 188x188 "pixels" of the matrix
        self.conv_2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1) # starts with 1 channel, will convert to {in_channels}x188x188
        self.conv_3 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1) # reduce channels to 64 ()
        self.conv_4 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, padding=1)
        self.flatten = nn.Flatten(start_dim=1) # makes completely linear from matrix
        self.fc_1 = nn.Linear(4000, 1000) # fully connected network
        self.fc_2 = nn.Linear(1000, 250)
        self.fc_3 = nn.Linear(250, 100) # fully connected network 
        self.fc_4 = nn.Linear(100, 25) # fully connected network
        self.fc_5 = nn.Linear(25, 1) # fully connected network
    
    def forward(self, x):
        
        x = self.conv_1(x)
        x = F.relu(x)

        x = self.conv_2(x)
        x = F.relu(x)
        
        x = self.conv_3(x)
        x = F.relu(x)

        x = self.conv_4(x)
        x = F.relu(x)
        # dimension conversion
        x = self.flatten(x) # change dimension from 3d to 1d, does not change numbers
        
        #apply fully connected layers 
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = F.relu(self.fc_4(x)) 
        x = self.fc_5(x) # reduces dimension from 64

        return x

    def initialize_weights(self, m):
        # parameter initialization
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

