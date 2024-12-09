#!/usr/bin/env python3

#
# Sample training script for GSnet
#
#  Sample data has been provided with 100 IDP structures
#  in both the training and test set for predicting the 
#  original 6 target values.
#

# Essential Imports
import os
import sys

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Torch Geometric
import torch_geometric
from torch_geometric.data import Dataset
from torch_geometric.nn import MessagePassing

# Project-Specific Modules
from dataset import ProteinDataset
from net import Net

def train(m,opt,microbatch=1):
    '''
    Trains GNN.

    Parameters
    ----------
    m          : network
    opt        : optimizer
    microbatch : (int, optional) Number of microbatches per batch.
    '''
    m.train()
    loss_sum = 0.0
    opt.zero_grad()

    for i, batch in enumerate(train_loader):
        batch.to(device)

        # Pass node features and connections
        pred = m(batch.pos,
                 batch.a,
                 batch.cc,
                 batch.dh,
                 batch.batch)
        # Calculate loss and gradients
        loss = loss_fn(pred, batch.y)
        loss.backward()
        
        # Update using the gradients
        if (i+1)%microbatch:
            opt.step()
            opt.zero_grad()
        loss_sum += loss

    return float( loss_sum / len(train_loader) )

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
                     batch.cc,
                     batch.dh,
                     batch.batch)
        
        # Calculate loss and gradients
        loss = loss_fn(pred, batch.y)
        loss_sum += loss
    return float( loss_sum / len(test_loader) )

# Load train data (using sample data for 100 IDP structures)
train_data = ProteinDataset(root='../sample_data/GSnet/train',
                           normalize=True,
                           use_dh=True,
                           use_cc=True,
                          )

# Load test data (using sample data for 100 IDP structures)
test_data = ProteinDataset(root='../sample_data/GSnet/test/',
                           avg=train_data.avg,
                           std=train_data.std,
                           normalize=False,
                           use_dh=True,
                           use_cc=True,
                          )

# Batch data
train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = torch_geometric.loader.DataLoader(test_data, batch_size=8, shuffle=False)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize network
net = Net(use_transfer      = True,      # Transfer learning
          out_channels_t    = 6,         # Number of output channels
          residue_pred      = False,     # Residue pred?
          hidden_channels   = 150,       # GNN Channels #150
          num_filters       = 150,       #              #150
          num_interactions  = 6,         # Number of GNN layers
          num_gaussians     = 300,       # Number of Gaussians
          cutoff            = 15.0,      # Cutoff (Å) for edges
          max_num_neighbors = 150,       # Max. edges per node # 150
          readout           = 'mean',    # Pooling method
          out_channels      = 6,         # Number of outputs (custom implementation)
          dropout           = 0.0,       # Dropout (Zero for no dropout)
          num_linear        = 4,         # Number of linear layers
          linear_channels   = 1024,      # Linear channels
          activation        = 'ssp',     # Linear activation
          cc_embedding      = 'rbf',     # CA-COFM Distance Embedding ('mlp', 'rbf')
          heads             = 1,         # Number of attention Heads (for transformerconv)
          mlp_activation    = 'relu',    # MLP Embedding Activation (relu, leakyrelu, ssp)
          standardize_cc    = True,      # Standardize CA-COFM distances?
          advanced_residual = True,      # More advanced residual?
          )

pt = torch.load('../models/GSnet_default.pt',map_location=device)
net.load_state_dict(pt,strict=False)

net = net.to(device)
sys.stderr.write(f'{net}\n')

for param in net.parameters():
    param.requires_grad = False # Set to true if optimizing the GNN itself

# Allow output MLP to optimize
for param in net.fc_t: # 'fc_t' is an MLP for transfer learning only
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
