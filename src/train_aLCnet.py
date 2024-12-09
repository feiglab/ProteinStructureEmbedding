#!/usr/bin/env python3

#
# Sample training script for aLCnet
#
#  Sample data has been provided with a few random stuctures
#  from the PHMD549 dataset (https://doi.org/10.1021/acs.jcim.3c00254)
#  for predicting pKa values.
#

## Imports
import os
import sys
import pickle
import os.path as osp
import warnings
from math import pi as PI
from typing import Optional
import time
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import mdtraj as md
import random
from scipy.stats import pearsonr
from time import strftime, localtime
import subprocess as sub

## Torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch_scatter import scatter

## Torch Geometric
import torch_geometric
from torch_geometric.data import Dataset, download_url, extract_zip
from torch_geometric.data.makedirs import makedirs
from torch_geometric.nn import MessagePassing, radius_graph

# 
from dataset import AtomicDataset
from net import Net_atomic as Net


def train(m,opt,microbatch=8):
    '''
    Trains GNN.

    Parameters
    ----------
    m          : network
    opt        : optimizer
    microbatch : (int, optional) Number of microbatches per batch.
    '''
    loss_sum = 0.0
    opt.zero_grad()

    m.train()

    for i, batch0 in enumerate(train_loader):

        # Pass node features and connections
        try:
            batch = batch0
            batch.to(device)

            pred = m(batch.pos,
                     batch.a,
                     batch.atom,
                     batch.charge,
                     batch=batch.batch,
                     resid_atomic=batch.resid_atomic,
                     resid_ca=batch.resid_ca,)

            # Calculate loss and gradients
            loss = loss_fn(pred, batch.y)
            loss.backward()
            
            # Update using the gradients
            if (i+1)%microbatch:
                opt.step()
                opt.zero_grad()
            loss_sum += loss
        
        except Exception as e:
            
            sys.stderr.write(f'Exception with {i} (training) \n-> {e}\n')


        return float( loss_sum / len(train_loader) )

# Validation
def validate(m):

    m.eval()

    loss_sum = 0.0
    for batch in test_loader:
        # Use GPU
        batch.to(device)
        # No grad
        with torch.no_grad():
            pred = m(batch.pos,
                     batch.a,
                     batch.atom,
                     batch.charge,
                     batch=batch.batch,
                     resid_atomic=batch.resid_atomic,
                     resid_ca=batch.resid_ca,)
        
        # Calculate loss and gradients
        loss = loss_fn(pred, batch.y)
        loss_sum += loss
    return float( loss_sum / len(test_loader) )


# Load train data (using sample data from PHMD549)
train_data = AtomicDataset(
    root='../sample_data/aLCnet/train',
    normalize=True
)

# Load test data (using sample data from PHMD549)
test_data = AtomicDataset(
    root='../sample_data/aLCnet/test',
    avg=train_data.avg,
    std=train_data.std,
    normalize=False,
)

# Batch data
train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = torch_geometric.loader.DataLoader(test_data, batch_size=1, shuffle=False)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize network
net = Net(
    fc_opt            = 1,      # Option for GNN -> FC “FCopt{N}”
    hidden_channels   = 75,     # Number of hidden dimensions -> “{N}channels”
    num_filters       = 150,    # Number of convolutional filters -> “{N}filters”
    num_interactions  = 3,      # Number of layers -> “{N}layers”
    num_gaussians     = 300,    # Number of gaussians for distance expansion
    sele_cutoff       = 10.0,   # Selection cutoff
    edge_cutoff       = 5.0,    # Radius graph cutoff -> “{X}edgecutoff”
    max_num_neighbors = 150,    # Maximum edges per node
    readout           = 'mean', # Read out the mean
    out_channels      = 1,      # Number of outcomes (1 for pKa)
    dropout           = 0.2,    # Dropout rate
    num_linear        = 6,      # Number of linear layers in FC -> “{N}lin”
    linear_channels   = 1024,   # Number of hidden linear dims -> “{N}FCch
    activation        = 'ssp',  # Activation function used in FC layers
    mlp_activation    = 'relu', # Activation function used in MLP embeddings
    heads             = 3,      # Number of transformer attention heads “{N}heads”
    advanced_residual = True,   # Create residual connections?
)

net = net.to(device)

# Load pretrained weights
state_dict = '../models/aLCnet_pKa.pt'
net.load_state_dict(torch.load(state_dict, map_location=device), strict=False)

for param in net.parameters():
    param.requires_grad = False # Set to true if optimizing the GNN itself

# Allow output MLP to optimize
for param in net.fc:
    try:
        param.weight.requires_grad = True
    except:
        pass
    try:
        param.bias.requires_grad = True
    except:
        pass

# Loss function
loss_fn = nn.MSELoss()
# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

best_loss = 100000.0
best_epoch = 0

for i in range(1001):
    net.train()
    loss = train(net,optimizer,microbatch=8)
    net.eval()
    loss_v = validate(net)

    if (loss_v < best_loss):
        torch.save(net.state_dict(), f'{i}epochs_state_dict.pt')
        best_epoch = i
        best_loss = loss_v

    # Write results
    sys.stderr.write(f'Epoch: {str(i).rjust(3)} | Training Loss: {str(round(loss,6)).rjust(10)} | Validation Loss: {str(round(loss_v,6)).rjust(10)}\n')
    
    # Terminate training if 10 consequtive epochs pass without improvement in validation performance
    if (i - best_epoch) > 10:
        sys.stderr.write('Training complete.\n')
        break

