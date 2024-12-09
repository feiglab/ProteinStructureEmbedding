#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Script to generate embeddings for PDBs in a directory, with protein- or residue-specific options.
#
# USAGE:
# python embed.py --protein/--residue PDBPATH OUTPATH
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

import sys
import torch
import os
import argparse
import multiprocessing as mp
from net import Net
from dataset import NumpyRep

def validate_arguments(args):
    """
    Validate the command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Raises
    ------
    ValueError
        If neither or both `--protein` and `--residue` are specified.
    """
    if not args.protein and not args.residue:
        raise ValueError("You must specify either --protein or --residue.")
    if args.protein and args.residue:
        raise ValueError("You cannot specify both --protein and --residue.")

def load_model(device, mode):
    """
    Load and initialize the neural network model based on the mode.

    Parameters
    ----------
    device : torch.device
        The device to run the model on (CPU or GPU).
    mode : str
        Either 'protein' or 'residue' to determine the model behavior.

    Returns
    -------
    net : Net
        The initialized neural network model.
    """
    if mode == "protein":
        net = Net(embedding_only=True,
                  hidden_channels=150,
                  num_filters=150,
                  num_interactions=6,
                  num_gaussians=300,
                  cutoff=15.0,
                  max_num_neighbors=150,
                  readout='mean',
                  activation='ssp',
                  cc_embedding='rbf',
                  heads=1,
                  mlp_activation='relu',
                  standardize_cc=True,
                  advanced_residual=True)
        state_dict = '../models/GSnet_default.pt'

    elif mode == "residue":
        net = Net(embedding_only=True,
                  residue_pred=True,
                  residue_pooling=False,
                  global_mean=True,
                  env_thresh=[6,8,10,12,15],
                  hidden_channels=150,
                  num_filters=150,
                  num_interactions=6,
                  num_gaussians=300,
                  cutoff=15.0,
                  max_num_neighbors=64,
                  readout='mean',
                  activation='ssp',
                  cc_embedding='rbf',
                  heads=1,
                  mlp_activation='relu',
                  standardize_cc=True,
                  advanced_residual=True)
        state_dict = '../models/GSnet_pKa.pt'

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
        #print(embedding.shape)
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

def main():
    """
    Main function to parse arguments, load the model, and process PDB files.
    """
    parser = argparse.ArgumentParser(description="Generate embeddings for PDBs using GSnet.")
    parser.add_argument("--protein", action="store_true", help="Use protein-specific embedding mode.")
    parser.add_argument("--residue", action="store_true", help="Use residue-specific embedding mode.")
    parser.add_argument("PDBPATH", type=str, help="Path to the directory containing PDB files.")
    parser.add_argument("OUTPATH", type=str, help="Path to the directory to save embeddings.")
    args = parser.parse_args()

    try:
        validate_arguments(args)
        mode = "protein" if args.protein else "residue"
        pdb_path = args.PDBPATH
        out_path = args.OUTPATH

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = load_model(device, mode)
        process_pdbs(pdb_path, out_path, net, device)

    except ValueError as ve:
        print(f"Argument Error: {ve}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

if __name__ == '__main__':
    main()

