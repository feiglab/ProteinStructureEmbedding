#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Simple script that generates embeddings for all PDBs in a directory and saves them in another directory.
#
# USAGE:
# python embed.py PDBPATH OUTPATH
#
# ------------------------------------------------------------------------------
# Citation Information
# ------------------------------------------------------------------------------
# Author: Spencer Wozniak
# Email: woznia79@msu.edu
# Paper Title: "Title of Your Research Paper"
# Published in: Journal Name, Volume, Page Numbers, Year
# DOI: https://doi.org/xxxxxxxx
# GitHub: https://github.com/yourusername/your-repository
#
# Please cite the above paper if you use this code in your research.
# ------------------------------------------------------------------------------

import sys
import torch
import pickle
import os
import multiprocessing as mp
from net import Net
from dataset import NumpyRep

def validate_arguments():
    """
    Validate the command line arguments.

    Raises
    ------
    ValueError
        If the number of arguments is not equal to 3.
    """
    if len(sys.argv) != 3:
        raise ValueError('USAGE: ./embed.py PDBPATH OUTPATH')

def load_model(device):
    """
    Load and initialize the neural network model with predefined parameters.

    Parameters
    ----------
    device : torch.device
        The device to run the model on (CPU or GPU).

    Returns
    -------
    net : Net
        The initialized neural network model.
    """
    net = Net(embedding_only=True,
              residue_pred=True,
              residue_pooling=False,
              global_mean=True,
              env_thresh=[6, 8, 10, 12, 15],
              hidden_channels=150,
              num_filters=150,
              num_interactions=6,
              num_gaussians=300,
              cutoff=15.0,
              max_num_neighbors=150,
              readout='mean',
              activation='ssp',
              cc_embedding='rbf',
              gnn_layer='transformerconv',
              heads=1,
              mlp_activation='relu',
              standardize_cc=True,
              layernorm=False,
              advanced_residual=True)

    model_dir = '../models'
    state_dict = '/feig/s1/spencer/gnn/main/models/pka_from_sasa_res.pt'
    dict1 = torch.load(state_dict, map_location=device)
    todel = [d for d in dict1 if 'fc' in d]

    for d in todel:
        del dict1[d]

    net.load_state_dict(dict1, strict=False)
    net.eval()
    return net

def create_embedding(pdb, net, device, out_path):
    """
    Create and save embedding for a given PDB file.

    Parameters
    ----------
    pdb : str
        The path to the PDB file.
    net : Net
        The neural network model.
    device : torch.device
        The device to run the model on (CPU or GPU).
    out_path : str
        The output directory to save the embeddings.
    """
    name = pdb.split('/')[-1].strip('.pdb')

    try:
        pdb_rep = NumpyRep(pdb)

        # Convert Numpy data to Torch tensors
        pos = torch.tensor(pdb_rep.x, dtype=torch.float).to(device)
        a = torch.tensor(pdb_rep.get_aas(), dtype=torch.long).to(device)
        cc = torch.tensor(pdb_rep.get_cc(), dtype=torch.float).reshape(-1, 1).to(device)
        dh = torch.tensor(pdb_rep.get_dh(), dtype=torch.float).to(device)

        # Get embedding and save
        embedding = net(pos, a, cc, dh)
        print(embedding.shape)
        torch.save(embedding, f'{out_path}/{name}.pt')
    except Exception as e:
        print(f'Error loading {pdb}: {e}')

def process_pdbs(pdb_path, out_path, net, device):
    """
    Process all PDB files in the given directory and generate embeddings.

    Parameters
    ----------
    pdb_path : str
        The input directory containing PDB files.
    out_path : str
        The output directory to save the embeddings.
    net : Net
        The neural network model.
    device : torch.device
        The device to run the model on (CPU or GPU).
    """
    pdbs = [f'{pdb_path}/{f}' for f in os.listdir(pdb_path)]
    with mp.Pool() as pool:
        pool.starmap(create_embedding, [(pdb, net, device, out_path) for pdb in pdbs])

def main(pdb_path, out_path):
    """
    Main function to load the model and process PDB files.

    Parameters
    ----------
    pdb_path : str
        The input directory containing PDB files.
    out_path : str
        The output directory to save the embeddings.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = load_model(device)
        process_pdbs(pdb_path, out_path, net, device)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    try:
        validate_arguments()
        pdb_path = sys.argv[1]
        out_path = sys.argv[2]
        main(pdb_path, out_path)
    except ValueError as ve:
        print(f"Argument Error: {ve}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

