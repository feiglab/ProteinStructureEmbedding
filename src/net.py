#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# # Neural network architectures used in our project
#
# ------------------------------------------------------------------------------
# Citation Information
# ------------------------------------------------------------------------------
# Authors: Spencer Wozniak, Giacomo Janson, Michael Feig
# Emails: spencerwozniak@gmail.com, mfeiglab@gmail.com
# Paper Title: "Accurate Predictions of Molecular Properties of Proteins via Graph Neural Networks and Transfer Learning"
# DOI: https://doi.org/xxxxxxxx
# GitHub: https://github.com/feiglab/hydropro_ml
#
# Please cite the above paper if you use this code in your research.
# ------------------------------------------------------------------------------


## Imports
import os
import sys
import os.path as osp
import warnings
from math import pi as PI
from typing import Optional
import time
import matplotlib.pyplot as plt
import random
import scipy
import numpy as np
from scipy.stats import pearsonr
from time import strftime, localtime
from typing import Optional, List, Union
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

## Torch
import torch
import torch_cluster
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch_scatter import scatter

## Torch Geometric
import torch_geometric
from torch_geometric.data import Dataset, download_url, extract_zip
from torch_geometric.data.makedirs import makedirs
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.nn import radius_graph
from torch_geometric.typing import Adj, Size, OptTensor, Tensor

class Net(nn.Module):
    """
    GNN architecture for GSnet.
    (with various configuration options for embeddings, layers, and output processing.)

    Parameters
    ----------
    hidden_channels : int, optional
        Number of hidden channels in GNN layers. Defaults to 150.
    num_filters : int, optional
        Number of filters in GNN layers. Defaults to 150.
    num_interactions : int, optional
        Number of GNN layers. Defaults to 6.
    num_gaussians : int, optional
        Number of gaussians used in distance expansion. Defaults to 300.
    cutoff : float, optional
        Cutoff (in angstrom) for edges. Defaults to 15.0.
    max_num_neighbors : int, optional
        Max number of edges per node. Defaults to 150.
    readout : str, optional
        Node pooling method. Defaults to 'mean'.
    out_channels : int, optional
        Number of outputs. Defaults to 6.
    use_transfer : bool, optional
        Enable transfer learning. Defaults to False.
    out_channels_t : int, optional
        Number of outputs for transfer learning. Defaults to 1.
    dropout : float, optional
        Dropout rate for linear layers. Defaults to 0.0.
    num_linear : int, optional
        Number of linear layers. Defaults to 4.
    linear_channels : int, optional
        Number of hidden channels in linear layers. Defaults to 1024.
    activation : str, optional
        Activation function for GNN layers ('ssp', 'relu'). Defaults to 'ssp'.
    heads : int, optional
        Number of heads for transformer layers. Defaults to 1.
    cc_embedding : str, optional
        Embedding type for CA-CofM distances ('mlp', 'rbf'). Defaults to 'rbf'.
    mlp_activation : str, optional
        Activation function for MLP layers ('relu', 'leakyrelu', 'ssp'). Defaults to 'relu'.
    standardize_cc : bool, optional
        Standardize CA-CofM distances. Defaults to True.
    advanced_residual : bool, optional
        Use advanced residual blocks. Defaults to True.
    residue_pred : bool, optional
        Enable residue-level predictions. Defaults to False.
    residue_pooling : bool, optional
        Pool residue information over all hidden channels. Defaults to False.
    global_mean : bool, optional
        Combine global mean with residue. Defaults to False.
    cc_gaussians : int, optional
        Number of gaussians for CC embedding. Defaults to 500.
    embedding_only : bool, optional
        Output GNN embedding only. Defaults to False.
    env_thresh : list, optional
        Get local environment embeddings (Å). Defaults to an empty list.
    one_hot_res : bool, optional
        Include one-hot encoding for amino acid type in fully connected layers. Defaults to False.
    env_mlp : bool, optional
        Run MLP across all environment features to reduce dimensionality. Defaults to False.

    Methods
    -------
    get_block(self, block, dim, advanced_residual: bool)
        Wraps the block in a ResidualBlock if advanced_residual is True.

    additional_init(self)
        Placeholder for additional initialization in child classes.

    reset_parameters(self)
        Resets parameters of the network.

    forward(self, pos, a, cc, dh, batch=None, resid=None)
        Handles embeddings of nodes and passes pos, embeddings to self._forward().

    _forward(self, pos, h, batch=None, resid=None, input_feats=None)
        Takes embedded inputs and operates via GNN layers + linear layers -> (out_channels) matrix.

    __repr__(self)
        Returns a string representation of the Net class.
    """

    def __init__(self, hidden_channels: int = 150, num_filters: int = 150,
                 num_interactions: int = 6, num_gaussians: int = 300,
                 cutoff: float = 15.0, max_num_neighbors: int = 150,
                 readout: str = 'mean', out_channels: int = 6,
                 use_transfer: bool = False, out_channels_t: int = 1,
                 dropout: float = 0.0, num_linear: int = 4,
                 linear_channels: int = 1024, activation: str = 'ssp',
                 heads: int = 1, cc_embedding: str = 'rbf', 
                 mlp_activation: str = 'relu', standardize_cc: bool = True, 
                 advanced_residual: bool = True, residue_pred: bool = False,
                 residue_pooling: bool = False, global_mean: bool = False,
                 cc_gaussians: int = 500,
                 embedding_only: bool = False, env_thresh: list = [],
                 one_hot_res: bool = False,
                 env_mlp: bool = False,):

        super().__init__()

        ## Assertions
        assert num_linear >= 2, f'Number of linear layers must be 2 or greater. ({num_linear} < 2))'
        assert activation in {'ssp', 'relu'}, f'{activation} is not a valid activation function'
        assert cc_embedding in {'mlp', 'rbf'}, f'{cc_embedding} is not a valid embedding type'
        assert mlp_activation in {'relu', 'leakyrelu', 'ssp'}, f'{mlp_activation} is not a valid MLP activation function'

        ## Attributes
        self.param = nn.Parameter(torch.empty(0))    # Dummy parameter (use 'Net.param' to see parameter attributes)
        self.last_file = None                        # Allows storing last file for debugging (e.g. `net.last_file = input_file)`
        self.hidden_channels = hidden_channels       # Number of hidden channels in GNN layers
        self.num_filters = num_filters               # Number of filters in GNN layers
        self.num_interactions = num_interactions     # Number of GNN layers
        self.num_gaussians = num_gaussians           # Number of gaussians used in distance expansion
        self.cutoff = cutoff                         # Cutoff (in angstrom) for edges
        self.max_num_neighbors = max_num_neighbors   # Max number of edges per node
        self.readout = readout                       # Node pooling method
        self.out_channels = out_channels             # Number of outputs
        self.use_transfer = use_transfer             # Enable transferred learning
        self.out_channels_t = out_channels_t         # Number of outputs (for transferred learning)
        self.dropout = dropout                       # Dropout for linear layers (0: no dropout)
        self.num_linear = num_linear                 # Number of linear layers
        if linear_channels is None:                  #
            linear_channels = hidden_channels // 2   #
        self.linear_channels = linear_channels       # Number of hidden channels in linear layers
        self.residue_pred = residue_pred             # Enable residue-level predictions (Need to set 'resid' in '_forward')
        self.residue_pooling = residue_pooling       # Pool residue information over all hidden channels (default=True)
        self.global_mean = global_mean               # Combine global mean with residue (cat)
        self.cc_embedding = cc_embedding             # Embedding type for CA-CofM distances
        self.standardize_cc = standardize_cc         # Standardize CA-CofM distances
        self.embedding_only = embedding_only         # Output GNN embedding only (No linear la
        self.env_thresh = env_thresh                 # Get local environment embeddings (Å)
        self.env_mlp = env_mlp                       # Run MLP across all env features to reduce dimensionality
        self.one_hot_res = one_hot_res               # Include one-hot encoding for aa type in FC

        ## Activation for GNN layers
        if activation == 'ssp':
            self.act = act = ShiftedSoftplus()
        elif activation == 'relu':
            self.act = act = nn.ReLU()

        ## Embedding of CA-CofM distances
        if cc_embedding == 'mlp':
            in_channels = 1
        elif cc_embedding == 'rbf':
            in_channels = cc_gaussians
            self.cc_rbf = GaussianSmearing(0.0, 150.0, cc_gaussians)

        ## Activation for MLP embedding
        if mlp_activation == 'relu':
            mlp_act = nn.ReLU()
        elif mlp_activation == 'leakyrelu':
            mlp_act = nn.LeakyReLU()
        elif mlp_activation == 'ssp':
            mlp_act = ShiftedSoftplus()

        ## Embeddings

        # Embedding of AA type
        self.embedding = Embedding(20, hidden_channels)

        # Embedding of CA-CofM distances
        self.embed_cc = nn.Sequential(nn.Linear(in_channels,self.hidden_channels),
                                      mlp_act,
                                      nn.Linear(self.hidden_channels, self.hidden_channels))

        # Embedding of dihedral angles
        self.embed_dh = nn.Sequential(nn.Linear(15,self.hidden_channels),
                                      mlp_act,
                                      nn.Linear(self.hidden_channels, self.hidden_channels))

        # Embedding of node from other embeddings
        self.embed_node = nn.Sequential(nn.Linear(self.hidden_channels*3, self.hidden_channels),
                                        mlp_act,
                                        nn.Linear(self.hidden_channels, self.hidden_channels))

        # Embedding of environmental features
        self.embed_env = nn.Sequential(nn.Linear(self.hidden_channels*len(env_thresh), self.hidden_channels),
                                       mlp_act,
                                       nn.Linear(self.hidden_channels, self.hidden_channels))

        # Embedding of edges
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        #
        def get_block(block, dim, advanced_residual: bool):
            if advanced_residual:
                return ResidualBlock(block, dim)
            else:
                return block

        # GNN Layers
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = CustomInteractionBlock(hidden_channels, num_gaussians,
                                           num_filters, heads)

            self.interactions.append(get_block(block, hidden_channels, advanced_residual))
        
        # Fully Connected Layers
        self.fc = ModuleList()
        self.fc.append(nn.Dropout(self.dropout))
        if residue_pred and not use_transfer and residue_pooling:
            in_ch = self.hidden_channels * (self.num_interactions + 1)
            self.fc.append(Linear(in_ch, linear_channels))
            self.fc.append(act)
        else:
            self.fc.append(Linear(hidden_channels, linear_channels))
            self.fc.append(act)
        for _ in range(self.num_linear - 2):
            self.fc.append(nn.Dropout(self.dropout))
            self.fc.append(Linear(linear_channels, linear_channels))
            self.fc.append(act)
        self.fc.append(nn.Dropout(self.dropout))
        self.fc.append(Linear(linear_channels, out_channels))

        # Transfer learning fc layers
        if env_thresh:
            if not env_mlp:
                in_ch = self.hidden_channels * ( len(self.env_thresh) + 2 )
            else:
                in_ch = self.hidden_channels * 3
        elif global_mean:
            in_ch = self.hidden_channels * 2
        else:
            in_ch = hidden_channels

        if one_hot_res:
            in_ch += 6

        self.fc_t = ModuleList()
        self.fc_t.append(nn.Dropout(self.dropout))
        self.fc_t.append(Linear(in_ch, linear_channels))
        self.fc_t.append(act)
        for _ in range(self.num_linear - 2):
            self.fc_t.append(nn.Dropout(self.dropout))
            self.fc_t.append(Linear(linear_channels, linear_channels))
            self.fc_t.append(act)
        self.fc_t.append(nn.Dropout(self.dropout))
        self.fc_t.append(Linear(linear_channels, out_channels_t))

        self.additional_init() # For child class

        self.reset_parameters()

    def additional_init(self):
        pass

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        for layer in self.fc:
            try:
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)
            except ValueError:
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
            except AttributeError:
                continue
        for layer in self.fc_t:
            if isinstance(layer, (nn.Linear, nn.Conv1d)):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)

    def forward(self, pos, a, cc, dh, batch=None, resid=None):
        """
        Handles embeddings of nodes. Passes pos, embeddings to self._forward() as (n,3), (n,d) matrices, respectively.

        Parameters
        ----------
        n = # of residues
        d = hidden_channels
        ----------
        pos      : (n,3) matrix
        a        : (n,20) matrix
        cc       : (n,1) matrix
        dh       : (n,15) matrix
        batch    : (optional) Used for batching
        resid    : (optional) Used for residue-level prediction (e.g. pKa)
        """

        assert a.dim() == 1 and a.dtype == torch.long
        batch = torch.zeros_like(a) if batch is None else batch

        h = self.embedding(a)

        if self.standardize_cc:
            mu    = 28.612717 # Mean
            sigma = 18.466433 # Std
            cc = ( cc - mu ) / sigma

        if self.cc_embedding == 'mlp':
            j = self.embed_cc(cc)
        elif self.cc_embedding == 'rbf':
            j = self.cc_rbf(cc)
            j = self.embed_cc(j)

        k = self.embed_dh(dh)

        h = self.embed_node(torch.cat([h,j,k], axis=1))

        if self.one_hot_res:
            input_feats = map_a_tensor(a[resid]).to(self.param.device)
        else:
            input_feats=None

        h = self._forward(pos, h, batch, resid, input_feats=input_feats)

        return h

    def _forward(self, pos, h, batch=None, resid=None, input_feats=None):
        """
        Takes embedded inputs and operates via GNN layers + linear layers -> (out_channels) matrix

        Parameters
        ----------
        n = # of residues
        d = hidden_channels
        ----------
        pos      : (n,3) matrix containing Cartesian coordinates of atoms
        h        : (n,d) matrix with embeddings passed from .forward() method
        batch    : (optional) Used for batching
        resid    : (optional) Mask that defines residue of interest
        """
        # Create edge indices using radius graph based on positions
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        # Extract row and column indices from edge_index
        row, col = edge_index
        # Calculate the Euclidean distance between nodes
        dists = (pos[row] - pos[col]).norm(dim=-1)

        edge_weight = dists
        edge_attr = self.distance_expansion(dists).to(self.param.device) # Gaussians

        if resid is not None and self.residue_pooling:
            h_aggr = []

        for i, interaction in enumerate(self.interactions):
            if resid is not None and self.residue_pooling:
                h_aggr.append(h[resid])

            h = h + interaction(h, edge_index, edge_weight, edge_attr) # Make EGNN.forward() take these same args.

        # Mean pooling
        if resid is None:
            h = scatter(h, batch, dim=0, reduce=self.readout)
        elif self.env_thresh:

            node_indices = torch.nonzero(resid).to(self.param.device)
            # compute global+local embeddings

            h_global = scatter(h, batch, dim=0, reduce=self.readout)
            h_local = h[resid]
            h_env = []
            for node_index in node_indices: # for each element in batch
                h_thresh = []
                for thresh in self.env_thresh: # for each threshold
                    sel_edge_thresh = edge_index[:,edge_weight < thresh] # select based on thresh
                    sel_edge_index = sel_edge_thresh[:,sel_edge_thresh[0] == node_index] # select based on node
                    h_mean = h[sel_edge_index[1]].mean(0,keepdims=True)
                    h_thresh.append(h_mean)
                h_thresh = torch.cat(h_thresh).unsqueeze(0)
                if self.env_mlp:
                    #print(h_thresh.shape)
                    h_thresh = self.embed_env(h_thresh.reshape(1,-1)).reshape(1,1,-1)
                h_env.append(h_thresh)
            h_env = torch.cat(h_env)
            h = torch.cat([h_local.unsqueeze(1),h_global.unsqueeze(1),h_env],dim=1)
            h = h.reshape(h.shape[0],-1)

        elif self.global_mean:
            h_global = scatter(h, batch, dim=0, reduce=self.readout)
            h = torch.cat([h[resid],h_global],axis=1)
        elif self.residue_pooling:
            h_aggr.append(h[resid])
            h = torch.cat(h_aggr,axis=1)
        else:
            h = h[resid]

        if input_feats is not None:
            input_feats = input_feats.to(self.param.device)
            h = torch.cat([input_feats,h],dim=1)

        if self.embedding_only:
            return h

        # Fully Connected
        if not self.use_transfer:
            for layer in self.fc:
                h = layer(h)
        else:
            for i, layer in enumerate(self.fc_t):
                h = layer(h)

        return h

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'num_linear={self.num_linear}, '
                f'linear_channels={self.linear_channels}, '
                f'max_num_neighbors={self.max_num_neighbors}, '
                f'cutoff={self.cutoff}, '
                f'activation={self.act}, '
                f'out_channels={self.out_channels}, '
                f'dropout={self.dropout}, '
                f'num_linear={self.num_linear}, '
                f'linear_channels={self.linear_channels}, '
                f'use_transfer={self.use_transfer}, '
                f'global_mean={self.global_mean}, '
                f'env_thresh={self.env_thresh}) ')

class Net_atomic(torch.nn.Module):
    """
    GNN architecture for a-GSnet.
    (Highly similar to GSnet)

    Parameters
    ----------
    hidden_channels : int, optional
        Number of hidden channels in GNN layers. Defaults to 150.
    num_filters : int, optional
        Number of filters in GNN layers. Defaults to 150.
    num_interactions : int, optional
        Number of GNN layers. Defaults to 4.
    num_gaussians : int, optional
        Number of gaussians used in distance expansion. Defaults to 300.
    sele_cutoff : float, optional
        Cutoff (in angstrom) for selection. Defaults to 10.0.
    edge_cutoff : float, optional
        Cutoff (in angstrom) for edges. Defaults to 5.0.
    max_num_neighbors : int, optional
        Max number of edges per node. Defaults to 150.
    readout : str, optional
        Node pooling method. Defaults to 'mean'.
    out_channels : int, optional
        Number of outputs. Defaults to 1.
    dropout : float, optional
        Dropout rate for linear layers. Defaults to 0.0.
    num_linear : int, optional
        Number of linear layers. Defaults to 4.
    linear_channels : int, optional
        Number of hidden channels in linear layers. Defaults to 1024.
    activation : str, optional
        Activation function for GNN layers ('ssp', 'relu'). Defaults to 'ssp'.
    heads : int, optional
        Number of heads for transformer layers. Defaults to 1.
    mlp_activation : str, optional
        Activation function for MLP layers ('relu', 'leakyrelu', 'ssp'). Defaults to 'relu'.
    advanced_residual : bool, optional
        Use advanced residual blocks. Defaults to True.
    global_mean : bool, optional
        Combine global mean with residue. Defaults to False.
    embedding_only : bool, optional
        Output GNN embedding only. Defaults to False.
    fc_opt : int, optional
        Option for fully connected layers. Defaults to 1.
    one_hot_res : bool, optional
        Include one-hot encoding for amino acid type in fully connected layers. Defaults to False.

    Methods
    -------
    additional_init(self)
        Placeholder for additional initialization in child classes.

    reset_parameters(self)
        Resets parameters of the network.

    forward(self, pos, a, atom, charge, resid_atomic=None, resid_ca=None, batch=None)
        Handles embeddings of nodes. Passes pos, embeddings to self._forward().

    _forward(self, pos, h, batch=None, resid_atomic=None, resid_ca=None, input_feats=None)
        Takes embedded inputs and operates via GNN layers + linear layers -> (out_channels) matrix.

    __repr__(self)
        Returns a string representation of the Net_atomic class.
    """
    def __init__(self, hidden_channels: int = 150, num_filters: int = 150,
                 num_interactions: int = 4, num_gaussians: int = 300,
                 sele_cutoff: float = 10.0, edge_cutoff: float = 5.0,
                 max_num_neighbors: int = 150, readout: str = 'mean',
                 out_channels: int = 1, dropout: float = 0.0,
                 num_linear: int = 4, linear_channels: int = 1024,
                 activation: str = 'ssp', heads: int = 1,
                 mlp_activation: str = 'relu',
                 advanced_residual: bool = True, global_mean: bool = False,
                 embedding_only: bool = False,
                 fc_opt: int = 1, one_hot_res: bool = False):

        super().__init__()

        ## Assertions
        assert num_linear >= 2, f'Number of linear layers must be 2 or greater. ({num_linear} < 2))'
        assert activation in {'ssp', 'relu'}, f'{activation} is not a valid activation function'
        assert mlp_activation in {'relu', 'leakyrelu', 'ssp'}, f'{mlp_activation} is not a valid MLP activation function'
        assert edge_cutoff <= sele_cutoff, 'Selection cutoff must be less than or equal to edge cutoff'
        assert fc_opt in {0, 1, 2, 3}, 'FC option must be in {0, 1, 2, 3}'

        ## Attributes
        self.param = nn.Parameter(torch.empty(0))
        self.last_file = None
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.edge_cutoff = edge_cutoff
        self.sele_cutoff = sele_cutoff
        self.max_num_neighbors = max_num_neighbors
        self.readout = readout
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_linear = num_linear
        self.linear_channels = linear_channels if linear_channels is not None else hidden_channels // 2
        self.global_mean = global_mean
        self.embedding_only = embedding_only
        self.fc_opt = fc_opt
        self.one_hot_res = one_hot_res

        ## Activation for GNN layers
        if activation == 'ssp':
            self.act = ShiftedSoftplus()
        elif activation == 'relu':
            self.act = nn.ReLU()

        ## Activation for MLP embedding
        if mlp_activation == 'relu':
            mlp_act = nn.ReLU()
        elif mlp_activation == 'leakyrelu':
            mlp_act = nn.LeakyReLU()
        elif mlp_activation == 'ssp':
            mlp_act = ShiftedSoftplus()

        ## Embeddings
        self.embedding_aa = Embedding(20, hidden_channels)
        self.embedding_atom = Embedding(5, hidden_channels)
        self.embed_charge = nn.Sequential(
            nn.Linear(1, self.hidden_channels),
            mlp_act,
            nn.Linear(self.hidden_channels, self.hidden_channels)
        )
        self.embed_node = nn.Sequential(
            nn.Linear(self.hidden_channels * 3, self.hidden_channels),
            mlp_act,
            nn.Linear(self.hidden_channels, self.hidden_channels)
        )
        self.distance_expansion = GaussianSmearing(0.0, edge_cutoff, num_gaussians)

        #
        def get_block(block, dim, advanced_residual: bool):
            if advanced_residual:
                return ResidualBlock(block, dim)
            else:
                return block

        # GNN Layers
        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = CustomInteractionBlock(hidden_channels, num_gaussians, num_filters, heads)
            self.interactions.append(get_block(block, hidden_channels, advanced_residual))

        # Get number of input layers to FC
        if fc_opt == 0:
            in_ch = hidden_channels
        elif fc_opt == 1:
            in_ch = hidden_channels * 3
        else:
            in_ch = hidden_channels * 2

        if one_hot_res:
            in_ch += 6

        # Fully Connected Layers
        self.fc = nn.ModuleList()
        self.fc.append(nn.Dropout(self.dropout))
        self.fc.append(nn.Linear(in_ch, linear_channels))
        self.fc.append(self.act)
        for _ in range(self.num_linear - 2):
            self.fc.append(nn.Dropout(self.dropout))
            self.fc.append(nn.Linear(linear_channels, linear_channels))
            self.fc.append(self.act)
        self.fc.append(nn.Dropout(self.dropout))
        self.fc.append(nn.Linear(linear_channels, out_channels))

        self.additional_init()
        self.reset_parameters()

    def additional_init(self):
        pass

    def reset_parameters(self):
        self.embedding_aa.reset_parameters()
        self.embedding_atom.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        for layer in self.fc:
            try:
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)
            except ValueError:
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
            except AttributeError:
                continue

    def forward(self, pos, a, atom, charge, resid_atomic=None, resid_ca=None, batch=None):
        """
        Handles embeddings of nodes. 
        Passes pos, embeddings to self._forward() as (n,3), (n,d) matrices, respectively.

        Parameters
        ----------
        pos          : (n,3) matrix
        a            : (n,1) matrix
        atom         : (n,1) matrix
        charge       : (n,1) matrix
        batch        : (optional) Used for batching
        resid_atomic : (optional) Used for residue-level prediction
        resid_ca     : (optional) Used for residue-level prediction
        """
        assert a.dim() == 1 and a.dtype == torch.long
        assert atom.dim() == 1 and atom.dtype == torch.long
        batch = torch.zeros_like(a) if batch is None else batch


        if len(charge.shape) == 1:
            charge = charge.unsqueeze(1)

        # Embed individual node feats
        i = self.embedding_aa(a)
        j = self.embedding_atom(atom)
        k = self.embed_charge(charge)

        # Embed node
        h = self.embed_node(torch.cat([i, j, k], axis=1))

        if self.one_hot_res:
            input_feats = map_a_tensor(a[resid_ca]).to(self.param.device)
        else:
            input_feats = None

        h = self._forward(pos, h, batch=batch, resid_atomic=resid_atomic, resid_ca=resid_ca, input_feats=input_feats)

        return h

    def _forward(self, pos, h, batch=None, resid_atomic=None, resid_ca=None, input_feats=None):
        """
        Passes embedded inputs to the GNN -> MLP to produce (out_channels) matrix.

        Parameters
        ----------
        pos          : (n,3) matrix containing Cartesian coordinates of atoms
        h            : (n,d) matrix with embeddings passed from .forward() method
        resid_atomic : (optional) Mask that defines atoms in residue of interest. 
        resid_ca     : (optional) Mask that defines alpha-carbon in residue of interest
        batch        : (optional) Used for batching
        """
        # Create edge indices using radius graph based on positions
        edge_index = radius_graph(pos, r=self.edge_cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        # Extract row and column indices from edge_index
        row, col = edge_index
        # Calculate the Euclidean distance between nodes
        dists = (pos[row] - pos[col]).norm(dim=-1)
        # Set the edge weight to the calculated distances
        edge_weight = dists
        # Expand the distances using a Gaussian expansion function to create edge attributes.
        edge_attr = self.distance_expansion(dists).to(self.param.device)

        # Loop through each layer of the GNN.
        for i, interaction in enumerate(self.interactions):
            # Residual connection pattern
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        if self.fc_opt == 0:
            h = scatter(h, batch, dim=0, reduce=self.readout)
        elif self.fc_opt == 1 and resid_atomic is not None and resid_ca is not None:
            h_global = scatter(h, batch, dim=0, reduce=self.readout)
            h = torch.cat([h[resid_ca], h[resid_atomic].mean(0).reshape(1, -1), h_global], dim=1)
            #print(h.shape)
        elif self.fc_opt == 2 and resid_atomic is not None:
            h = h[resid_atomic].mean(0).reshape(1, -1)
        elif self.fc_opt == 3 and resid_ca is not None:
            h = h[resid_ca]
        else:
            raise RuntimeError('FC option does not match input to Net.forward()')

        if input_feats is not None:
            h = torch.cat([input_feats.to(self.param.device), h], dim=1)

        if self.embedding_only:
            return h

        for layer in self.fc:
            h = layer(h)

        return h

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'edge_cutoff={self.edge_cutoff}, '
                f'sele_cutoff={self.sele_cutoff}, '
                f'out_channels={self.out_channels})')

class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) with multiple linear layers and activation functions.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels. Defaults to 2560.
    num_linear : int, optional
        Number of linear layers. Defaults to 6.
    linear_channels : int, optional
        Number of hidden channels in linear layers. Defaults to 1024.
    dropout : float, optional
        Dropout rate for linear layers. Defaults to 0.2.
    out_channels : int, optional
        Number of output channels. Defaults to 1.

    Methods
    -------
    reset_parameters(self)
        Resets parameters of the network.

    forward(self, *inputs)
        Forward pass through the network. Concatenates inputs if multiple tensors are provided.
    """
    def __init__(self, in_channels: int = 2560,
                 num_linear: int = 6,
                 linear_channels: int = 1024,
                 dropout: float = 0.2,
                 out_channels: int = 1):
        super().__init__()

        # Activation function
        act = ShiftedSoftplus()
        self.in_channels = in_channels

        # Fully Connected Layers
        self.fc = ModuleList()
        self.fc.append(nn.Dropout(dropout))  # Dropout layer
        self.fc.append(Linear(in_channels, linear_channels))  # First linear layer
        self.fc.append(act)  # Activation function

        # Intermediate linear layers with dropout and activation
        for _ in range(num_linear - 2):
            self.fc.append(nn.Dropout(dropout))
            self.fc.append(Linear(linear_channels, linear_channels))
            self.fc.append(act)

        self.fc.append(nn.Dropout(dropout))  # Final dropout layer
        self.fc.append(Linear(linear_channels, out_channels))  # Output linear layer

    def reset_parameters(self):
        """
        Resets parameters of the network.
        Initializes weights using Xavier uniform distribution and biases to zero.
        """
        for layer in self.fc:
            try:
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)
            except ValueError:
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
            except AttributeError:
                continue

    def forward(self, *inputs):
        """
        Forward pass through the network. Concatenates inputs if multiple tensors are provided.

        Parameters
        ----------
        *inputs : torch.Tensor
            Input tensors to be passed through the network.

        Returns
        -------
        torch.Tensor
            Output of the network.
        """
        # Concatenate multiple input tensors if provided
        if len(inputs) > 1:
            x = torch.cat(inputs, dim=1)
        else:
            x = inputs[0]

        # Ensure the input tensor has the correct shape
        if x.dim() != 2 or x.size(1) != self.in_channels:
            raise ValueError(f"Expected input tensor to have shape [1, {self.in_channels}], got {x.shape}")

        # Pass input through each layer in the fully connected network
        for layer in self.fc:
            x = layer(x)

        return x


class FC(MLP):
    """
    Fully Connected (FC) network.
    (Alias for 'MLP' class)
    """
    pass


def map_a_tensor(input_tensor):
    """
    Maps an input tensor of specific values to a one-hot encoded tensor.

    The mapping is defined by the `a_to_aa` dictionary, where the keys are the
    input values, and the values are the corresponding one-hot encoded indices.

    Args:
        input_tensor (torch.Tensor): Input tensor containing values to be mapped.

    Returns:
        torch.Tensor: A tensor with one-hot encoding based on the mapping.
    """
    a_to_aa = {
        10: 0,
        2: 1,
        13: 2,
        14: 3,
        5: 4,
        16: 5
    }

    num_classes = len(a_to_aa)
    mapped_tensor = torch.zeros((num_classes, len(input_tensor)))

    for i, value in enumerate(input_tensor):
        mapped_value = a_to_aa[int(value.item())]
        mapped_tensor[mapped_value, i] = 1

    return mapped_tensor.t()


def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32, batch_size=None):
    """
    Custom 'radius' implementation to find neighbor pairs within a specified radius.
    (Needed to prevent)

    Parameters
    ----------
    x : torch.Tensor
        Source coordinates with shape (N, D) where N is the number of points and D is the dimensionality.
    y : torch.Tensor
        Target coordinates with shape (M, D) where M is the number of points and D is the dimensionality.
    r : float
        Radius within which to search for neighbors.
    batch_x : torch.Tensor, optional
        Batch indices for x. Default is None.
    batch_y : torch.Tensor, optional
        Batch indices for y. Default is None.
    max_num_neighbors : int, optional
        Maximum number of neighbors to return for each point in y. Default is 32.
    batch_size : int, optional
        Batch size for processing. Default is None.

    Returns
    -------
    torch.Tensor
        Tensor of shape (2, E) where E is the number of edges (neighbor pairs).
        The first row contains the indices of the source points, and the second row contains the indices of the target points.
    """
    # Return empty tensor if there are no elements in x or y
    if x.numel() == 0 or y.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    # Ensure x and y are at least 2-dimensional
    x = x.unsqueeze(1) if x.dim() == 1 else x
    y = y.unsqueeze(1) if y.dim() == 1 else y

    # Compute squared distances using broadcasting
    dists_squared = torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=-1)

    # Get the indices of distances that are less than r^2
    row, col = (dists_squared < r ** 2).nonzero(as_tuple=True)

    # Enforce the max_num_neighbors constraint
    row_list, col_list = [], []
    for i in range(y.size(0)):
        neighbors = (row == i).nonzero(as_tuple=True)[0]
        neighbors = neighbors[:max_num_neighbors]
        row_list.extend(row[neighbors].tolist())
        col_list.extend(col[neighbors].tolist())

    # Convert lists to tensors
    row = torch.tensor(row_list, dtype=torch.long, device=x.device)
    col = torch.tensor(col_list, dtype=torch.long, device=x.device)

    # Stack the row and col tensors to create the result
    result = torch.stack([row, col], dim=0)

    return result


def radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32, flow='source_to_target', batch_size=None):
    """
    Custom 'radius_graph' implementation to construct a graph where edges connect nodes within a specified radius.

    Parameters
    ----------
    x : torch.Tensor
        Node coordinates with shape (N, D) where N is the number of nodes and D is the dimensionality.
    r : float
        Radius within which to search for neighboring nodes.
    batch : torch.Tensor, optional
        Batch indices for x. Default is None.
    loop : bool, optional
        If True, self-loops are included in the graph. Default is False.
    max_num_neighbors : int, optional
        Maximum number of neighbors to return for each node. Default is 32.
    flow : str, optional
        Direction of message passing ('source_to_target' or 'target_to_source'). Default is 'source_to_target'.
    batch_size : int, optional
        Batch size for processing. Default is None.

    Returns
    -------
    torch.Tensor
        Tensor of shape (2, E) where E is the number of edges (neighbor pairs).
        The first row contains the indices of the source nodes, and the second row contains the indices of the target nodes.
    """
    assert flow in ['source_to_target', 'target_to_source'], "flow must be 'source_to_target' or 'target_to_source'"

    # Set batch_size if batch is not None
    if batch is not None:
        batch_size = 8

    # Get edge indices within the specified radius
    edge_index = radius(x, x, r, batch, batch, max_num_neighbors if loop else max_num_neighbors + 1, batch_size)

    # Adjust edge direction based on flow
    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]

    # Remove self-loops if loop is False
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]

    # Stack the row and col tensors to create the result
    result = torch.stack([row, col], dim=0)

    return result


class ResidualBlock(nn.Module):
    """
    Residual GNN block.

    This class defines a residual block for a Graph Neural Network (GNN).
    It includes a layer normalization step followed by a ReLU activation.

    Parameters
    ----------
    block : nn.Module
        The neural network block to apply within the residual block.
    dim : int
        The dimension of the input and output features.

    Methods
    -------
    reset_parameters():
        Resets the parameters of the block.
    forward(x, edge_index, edge_weight, edge_attr):
        Forward pass through the residual block.
    """
    def __init__(self, block, dim):
        super(ResidualBlock, self).__init__()
        self.block = block  # The neural network block to apply
        self.layernorm = nn.LayerNorm(dim)  # Layer normalization
        self.act = nn.ReLU()  # ReLU activation

    def reset_parameters(self):
        """
        Resets the parameters of the block.
        """
        self.block.reset_parameters()

    def forward(self, x, edge_index, edge_weight, edge_attr):
        """
        Forward pass through the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input node features.
        edge_index : torch.Tensor
            Graph edge indices.
        edge_weight : torch.Tensor
            Edge weights.
        edge_attr : torch.Tensor
            Edge attributes.

        Returns
        -------
        torch.Tensor
            Output node features after applying the residual block.
        """
        # Apply the block to the input features and edge data
        h = self.block(x, edge_index, edge_weight, edge_attr)
        # Apply layer normalization
        h = self.layernorm(h)
        # Apply ReLU activation
        h = self.act(h)

        return h

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'block={self.block}, '
                f'layernorm={self.layernorm}, '
                f'act={self.act}) ')


class InteractionBlock(nn.Module):
    """
    Interaction block for a GNN with continuous-filter convolution.

    Parameters
    ----------
    hidden_channels : int
        Number of hidden channels in GNN layers.
    num_gaussians : int
        Number of gaussians used in distance expansion.
    num_filters : int
        Number of filters in GNN layers.
    cutoff : float
        Cutoff (in angstrom) for edges.

    Methods
    -------
    __init__(self, hidden_channels, num_gaussians, num_filters, cutoff)
        Initializes the InteractionBlock with the given parameters.

    reset_parameters(self)
        Resets parameters of the network.

    forward(self, x, edge_index, edge_weight, edge_attr)
        Forward pass through the interaction block.
    """
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets parameters of the network. Initializes weights using Xavier uniform distribution and biases to zero.
        """
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        """
        Forward pass through the interaction block.

        Parameters
        ----------
        x : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Edge indices.
        edge_weight : torch.Tensor
            Edge weights.
        edge_attr : torch.Tensor
            Edge attributes.

        Returns
        -------
        torch.Tensor
            Updated node features.
        """
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CustomInteractionBlock(nn.Module):
    """
    Custom interaction block for a GNN using TransformerConv.

    Parameters
    ----------
    hidden_channels : int
        Number of hidden channels in GNN layers.
    num_gaussians : int
        Number of gaussians used in distance expansion.
    num_filters : int
        Number of filters in GNN layers.
    heads : int
        Number of attention heads in TransformerConv.

    Methods
    -------
    __init__(self, hidden_channels, num_gaussians, num_filters, heads)
        Initializes the CustomInteractionBlock with the given parameters.

    reset_parameters(self)
        Resets parameters of the network.

    forward(self, x, edge_index, edge_weight, edge_attr)
        Forward pass through the custom interaction block.
    """
    def __init__(self, hidden_channels, num_gaussians, num_filters, heads):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = TransformerConv(hidden_channels, hidden_channels // heads,
                                    heads, edge_dim=num_filters)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets parameters of the network. Initializes weights using Xavier uniform distribution and biases to zero.
        """
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        """
        Forward pass through the custom interaction block.

        Parameters
        ----------
        x : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Edge indices.
        edge_weight : torch.Tensor
            Edge weights.
        edge_attr : torch.Tensor
            Edge attributes.

        Returns
        -------
        torch.Tensor
            Updated node features.
        """
        e = self.mlp(edge_attr)

        x = self.conv(x, edge_index, e)
        x = self.act(x)
        x = self.lin(x)

        return x

class GaussianSmearing(torch.nn.Module):
    """
    Gaussian smearing of interatomic distances.

    Parameters
    ----------
    start : float, optional
        The start value for the Gaussian functions. Defaults to 0.0.
    stop : float, optional
        The stop value for the Gaussian functions. Defaults to 5.0.
    num_gaussians : int, optional
        The number of Gaussian functions. Defaults to 50.

    Methods
    -------
    forward(self, dist)
        Applies Gaussian smearing to the input distances.
    """
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        """
        Applies Gaussian smearing to the input distances.

        Parameters
        ----------
        dist : torch.Tensor
            Tensor of distances.

        Returns
        -------
        torch.Tensor
            Tensor of Gaussian-smearing values.
        """
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
    def __repr__(self):
        return f"GaussianSmearing(start={self.offset[0].item()}, stop={self.offset[-1].item()}, num_gaussians={len(self.offset)})"


class ShiftedSoftplus(torch.nn.Module):
    """
    Shifted Softplus activation function.

    Methods
    -------
    forward(self, x)
        Applies the Shifted Softplus activation function to the input tensor.
    """
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        """
        Applies the Shifted Softplus activation function to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying the Shifted Softplus activation.
        """
        return F.softplus(x) - self.shift

    def __repr__(self):
        return "ShiftedSoftplus()"


def main():

    # Parameters used for original network training
    model1 = Net(
          use_transfer      = False,
          hidden_channels   = 150,       # GNN Channels #150
          num_filters       = 150,       #              #150
          num_interactions  = 6,         # Number of GNN layers
          num_gaussians     = 300,       # Number of Gaussians
          cutoff            = 15.0,      # Cutoff (Å) for edges
          max_num_neighbors = 150,       # Max. edges per node # 150
          readout           = 'mean',    # Pooling method
          out_channels      = 1,         # Number of outputs (custom implementation)
          dropout           = 0.2,       # Dropout (Zero for no dropout)
          num_linear        = 4,         # Number of linear layers
          linear_channels   = 1024,      # Linear channels
          activation        = 'ssp',     # Linear activation
          cc_embedding      = 'rbf',     # CA-COFM Distance Embedding ('mlp', 'rbf')
          heads             = 1,         # Heads (for transformerconv)
          mlp_activation    = 'relu',    # MLP Embedding Activation (relu, leakyrelu, ssp)
          standardize_cc    = True,      # Standardize CA-COFM distances?             
          advanced_residual = True,     # More advanced residual?
    )
    
    # Parameters used for fine-tuning on molecular SASA
    model2 = Net(
          use_transfer      = True,
          out_channels_t    = 1,
          residue_pred      = False,
          hidden_channels   = 150,       # GNN Channels #150
          num_filters       = 150,       #              #150
          num_interactions  = 6,         # Number of GNN layers
          num_gaussians     = 300,       # Number of Gaussians
          cutoff            = 15.0,      # Cutoff (Å) for edges
          max_num_neighbors = 150,       # Max. edges per node # 150
          readout           = 'mean',    # Pooling method
          out_channels      = 1,         # Number of outputs (custom implementation)
          dropout           = 0.2,       # Dropout (Zero for no dropout)
          num_linear        = 4,         # Number of linear layers
          linear_channels   = 1024,      # Linear channels
          activation        = 'ssp',     # Linear activation
          cc_embedding      = 'rbf',     # CA-COFM Distance Embedding ('mlp', 'rbf')
          heads             = 1,         # Heads (for transformerconv)
          mlp_activation    = 'relu',    # MLP Embedding Activation (relu, leakyrelu, ssp)
          standardize_cc    = True,      # Standardize CA-COFM distances?             
          advanced_residual = True,     # More advanced residual?
    )
    
    # Parameters used for fine-tuning on pKa
    model3 = Net(
          use_transfer      = True,
          out_channels_t    = 1,
          residue_pred      = True,
          global_mean       = True,
          env_thresh        = [6,8,10,12,15],
          env_mlp           = False,
          one_hot_res       = True,
          hidden_channels   = 150,  # 64, 96, 128
          num_filters       = 150,   # 128
          num_interactions  = 6,         # 6
          num_gaussians     = 300,       # 200
          cutoff            = 15.0,      # 15.0
          max_num_neighbors = 64,        # 32, 56, 64
          readout           = 'mean',    # 'mean'
          out_channels      = 1,         # Number of outputs (custom implementation)
          dropout           = 0.2,
          num_linear        = 6,
          linear_channels   = 1024,
          activation        = 'ssp',
          cc_embedding      = 'rbf',
          heads             = 1,           # Need to set if using 'transformerconv'
          advanced_residual = True,
         )

    # Parameters used for training aGSnet on pKa
    model4 = Net_atomic(
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
          one_hot_res       = False,  # Append one-hot residue encoding to FC?
    )
         
    for model in (model1, model2, model3, model4):
        print(model)

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f'{total} total parameters.')
        print(f'{trainable} trainable parameters.')


if __name__ == '__main__':
    main()

