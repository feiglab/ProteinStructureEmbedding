#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Script for making predictions
#
# ------------------------------------------------------------------------------
# Citation Information
# ------------------------------------------------------------------------------
# Authors: Spencer Wozniak, Giacomo Janson, Michael Feig
# Emails: spencerwozniak1@gmail.com, mfeiglab@gmail.com
# Paper Title: "Accurate Predictions of Molecular Properties of Proteins via Graph Neural Networks and Transfer Learning"
# DOI: https://doi.org/10.1101/2024.12.10.627714
# GitHub: https://github.com/feiglab/ProteinStructureEmbedding
#
# Please cite the above paper if you use this code in your research.
# ------------------------------------------------------------------------------

import argparse
import mdtraj as md
import numpy as np
import torch
import torch_geometric
import torch.multiprocessing as mp
from torch_scatter import scatter
import os
import sys
import warnings
import subprocess as sub
from collections import OrderedDict
from Bio import PDB
from time import perf_counter
#warnings.filterwarnings('ignore')

from dataset import NumpyRep, NumpyRep_atomic
from net import Net, Net_atomic, ShiftedSoftplus, FC

#print("PyTorch version:", torch.__version__)
#print("PyTorch Geometric version:", torch_geometric.__version__)
#print("CUDA version:", torch.version.cuda)

class AtomSelect(PDB.Select):
    """
    Class for cleaning input PDB files by filtering out unwanted atoms and residues.

    Methods
    -------
    accept_atom(atom)
        Determines whether an atom should be accepted or rejected based on its type.

    accept_residue(residue)
        Determines whether a residue should be accepted or rejected based on its name.
    """

    def accept_atom(self, atom):
        """
        Accepts or rejects an atom based on its type.

        Parameters
        ----------
        atom : Bio.PDB.Atom
            The atom to evaluate.

        Returns
        -------
        bool
            True if the atom should be accepted, False otherwise.
        """
        # Reject atoms that are of type 'HETATM' or 'ANISOU'
        if atom.get_fullname().strip()[0:6] in ['HETATM', 'ANISOU']:
            return False
        return True

    def accept_residue(self, residue):
        """
        Accepts or rejects a residue based on its name.

        Parameters
        ----------
        residue : Bio.PDB.Residue
            The residue to evaluate.

        Returns
        -------
        bool
            True if the residue should be accepted, False otherwise.
        """
        # List of standard amino acids
        standard_aas = ['GLN', 'TRP', 'GLU', 'ARG', 'THR', 'TYR', 'ILE', 'PRO',
                        'ALA', 'SER', 'ASP', 'PHE', 'GLY', 'HIS', 'LYS', 'LEU',
                        'CYS', 'VAL', 'ASN', 'MET', 'HSD', 'HSE', 'HSP']

        # Accept only standard amino acids
        if residue.get_resname() not in standard_aas:
            return False
        return True


def parse_args():
    """
    Parse command-line arguments for the ML prediction script on PDB files.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.

    Raises
    ------
    RuntimeError
        If there is an error parsing arguments.
    """
    parser = argparse.ArgumentParser(description='ML prediction on PDB files')
    parser.add_argument('--clean', action='store_true', help='Clean PDB files before making predictions.')
    parser.add_argument('--pka', action='store_true', help='Predict pKa.')
    parser.add_argument('--atomic', action='store_true', help='Use aLCnet for pKa predictions')
    parser.add_argument('--sasa', action='store_true', help='Predict SASA.')
    parser.add_argument('--shift', action='store_true', help='Calculate pKa shift (relative to standard value).')
    parser.add_argument('--chain', metavar='chain', help='Specify chain.')
    parser.add_argument('--combine-chains', action='store_true', help='Make calculation for structure of all chains in a PDB file.')
    parser.add_argument('--keep', action='store_true', help='Keep cleaned PDB files.')
    parser.add_argument('--cpu', action='store_true', help='Run on CPU.')
    parser.add_argument('--gpu', action='store_true', help='Run on GPU.')
    parser.add_argument('--numpy', action='store_true', help='Use .npz file as input.')
    parser.add_argument('--time', action='store_true', help='Time different aspects of the model.')
    parser.add_argument('--skip-bad-files', action='store_true', help='Skip bad PDB files.')
    parser.add_argument('pdbs', nargs='+', help='List of PDB files.')

    args = parser.parse_args()

    # Assertions to ensure valid argument combinations
    assert not (args.cpu and args.gpu), 'Cannot run on both CPU and GPU!'
    assert not ((args.chain is not None) and args.combine_chains), f'Cannot select chain {args.chain} and combine chains.'
    assert not (args.sasa and args.pka), 'Cannot calculate SASA when predicting pKa.'
    
    if args.shift:
        assert args.pka, "'--pka' must be selected to calculate shifts."
    
    if args.atomic:
        assert args.pka, "'--pka' must be selected to use a-GSnet"

    if args.pka and (not args.atomic):
        sys.stderr.write("##### WARNING! #####\n"
                         "You are currently running pKa predictions with GSnet.\n"
                         "GSnet is slower and less accurate than a-GSnet for pKa predictions.\n"
                         "Use --atomic to run pKa predictions with the faster, more accurate a-GSnet!\n"
                         "(Ignore this warning if you are intentionally using GSnet.)\n"
                         "####################\n"
                         )

    try:
        return args
    except Exception as e:
        raise RuntimeError(f'Error parsing arguments: {e}')


def load_models(args, device):
    """
    Load ML models based on the provided arguments.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments parsed by `argparse.ArgumentParser().parse_args()`.
    device : torch.device
        Specifies the device on which computations will be performed. ("cuda" -> GPU / "cpu" -> CPU)

    Returns
    -------
    dict
        Dictionary containing the loaded models.

    Raises
    ------
    AssertionError
        If both `--pka` and `--sasa` arguments are selected.
    """
    # Ensure --pka and --sasa are not both selected
    assert not (args.pka and args.sasa), "Both --pka and --sasa cannot be selected"
    
    # Timing model loading if specified
    if args.time:
        t0 = perf_counter()
    
    model_dir = '../models'
    models = {}

    if args.pka:
        if not args.atomic:
            models['gnn'] = Net(use_transfer=True, out_channels_t=1, residue_pred=True,
                                global_mean=True, env_thresh=[6,8,10,12,15],
                                one_hot_res=True, hidden_channels=150,
                                num_filters=150, num_interactions=6, num_gaussians=300,
                                cutoff=15.0, max_num_neighbors=64, readout='mean',
                                out_channels=1, dropout=0.0, num_linear=6, linear_channels=1024,
                                activation='ssp', cc_embedding='rbf',
                                heads=1, mlp_activation='relu', standardize_cc=True,
                                advanced_residual=True)

            state_dict = torch.load(f'{model_dir}/GSnet_pKa.pt', map_location=device)
            models['gnn'].load_state_dict(state_dict, strict=False)
        else:
            models['gnn'] = Net_atomic(
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
                              dropout           = 0.0,    # Dropout rate
                              num_linear        = 6,      # Number of linear layers in FC -> “{N}lin”
                              linear_channels   = 1024,   # Number of hidden linear dims -> “{N}FCch
                              activation        = 'ssp',  # Activation function used in FC layers
                              mlp_activation    = 'relu', # Activation function used in MLP embeddings
                              heads             = 3,      # Number of transformer attention heads “{N}heads”              
                              advanced_residual = True,   # Create residual connections?
                              one_hot_res       = False,  # Append one-hot residue encoding to FC?
                            )
            #state_dict = torch.load(f'{model_dir}/aLCnet_pKa.pt', map_location=device)
            state_dict = torch.load(f'{model_dir}/aLCnet_pKa.pt', map_location=device)

            models['gnn'].load_state_dict(state_dict, strict=False)
    elif args.sasa:
        models['gnn'] = Net(embedding_only=True)
        state_dict = torch.load(f'{model_dir}/GSnet_default.pt', map_location=device)
        models['gnn'].load_state_dict(state_dict, strict=False)

        models['sasa'] = FC(in_channels=150, num_linear=4, out_channels=1, dropout=0.0)
        fc_elements = OrderedDict((f'fc{k[4:]}', v) for k, v in torch.load(f'{model_dir}/GSnet_SASA.pt', map_location=device).items() if k.startswith('fc_t'))
        models['sasa'].load_state_dict(fc_elements)

        models['6'] = FC(in_channels=150, num_linear=4, out_channels=6, dropout=0.0)
        fc_elements = OrderedDict((k, v) for k, v in state_dict.items() if k.startswith('fc.'))
        models['6'].load_state_dict(fc_elements)
    else:
        models['gnn'] = Net()
        models['gnn'].load_state_dict(torch.load(f'{model_dir}/GSnet_default.pt', map_location=device), strict=False)

    # Set models to evaluation mode and move to device
    for key in models.keys():
        if key != 'batch_converter':
            models[key].eval()
            models[key] = models[key].to(device)

    if args.time:
        t1 = perf_counter()
        print(f'LOADING MODEL: {t1-t0} s')

    return models


def print_result(calc_type, pred, pdb, chain=None, resid=None, aa=None):
    """
    Display prediction(s) based on the type of calculation.

    Parameters
    ----------
    calc_type : str
        Type of calculation. This should be one of the following:
            'pka'    : Display pKa predictions.
            'sasa'   : Display global predictions with SASA.
            'default': Display global predictions without SASA.
    pred : float or np.ndarray
        Output of ML model.
        If 'pka' is chosen, 'pred' will be a float.
        If 'sasa'/'default' is chosen, 'pred' will be a numpy array.
    pdb : str
        Path to the file that prediction is being made on.
    chain : str, optional
        Chain ID. Used when there are predictions on multiple chains.
    resid : int, optional
        Index of residue on which prediction is being made.
        Only used when 'pka' is chosen.
    aa : str, optional
        Amino acid type (3-letter abbreviation).
        Only used when 'pka' is chosen.
    """
    if calc_type == 'pka':
        if chain is None:
            print(pred, aa, resid, pdb)
        else:
            print(pred, aa, resid, chain, pdb)
    elif calc_type == 'sasa':
        if chain is None:
            print(f'%.6E %.7E %.7E %.7E %.7E %.7E %.7E {pdb}' % (pred[5], pred[0], pred[4], pred[1], pred[2], pred[3], pred[6]))
        else:
            print(f'%.6E %.7E %.7E %.7E %.7E %.7E %.7E {chain} {pdb}' % (pred[5], pred[0], pred[4], pred[1], pred[2], pred[3], pred[6]))
    else:
        if chain is None:
            print(f'%.6E %.7E %.7E %.7E %.7E %.7E {pdb}' % (pred[5], pred[0], pred[4], pred[1], pred[2], pred[3]))
        else:
            print(f'%.6E %.7E %.7E %.7E %.7E %.7E {chain} {pdb}' % (pred[5], pred[0], pred[4], pred[1], pred[2], pred[3]))


def create_representation(pdb, args):
    """
    Create Numpy representation(s) of PDB file.

    Parameters
    ----------
    pdb : str
        Path to the PDB file that prediction is being made on.
    args : argparse.Namespace
        The command line arguments parsed by `argparse.ArgumentParser().parse_args()`.

    Returns
    -------
    list
        A list of Numpy representations of the PDB file.
    """

    if args.time:
        t0 = perf_counter()

    if not args.numpy:
        try: # Try to create a numpy representation of the PDB
            pdb_rep = NumpyRep(pdb)
        except FileNotFoundError:
            raise FileNotFoundError(f'{pdb} not found.')
        except Exception as e:
            if args.skip_bad_files:
                print(f'Problem with {pdb}')
                return []
            warnings.warn('Initial PDB representation failed. Attempting to clean PDB file.')
            try:
                new = f'{pdb[:-4]}_new.pdb'
                io = PDB.PDBIO()
                io.set_structure(PDB.PDBParser().get_structure(None, pdb))
                io.save(new, select=AtomSelect())
                pdb_rep = create_representation(new, args)[0]
                if not args.keep and not args.atomic:
                    os.remove(new)
            except Exception as e:
                raise RuntimeError(f'Error creating representation from {pdb}: {e}')

        reps = [pdb_rep]

        if pdb_rep.traj.n_chains != 1:
            structure = PDB.PDBParser().get_structure(None, pdb)
            io = PDB.PDBIO()

            if not args.combine_chains:
                if args.chain is not None:
                    new = f'{pdb[:-4]}_{args.chain}.pdb'
                    io.set_structure(structure[0][args.chain])
                    io.save(new, select=AtomSelect())
                    pdb_rep = create_representation(new, args)[0]
                    pdb_rep.chain = args.chain
                    reps = [pdb_rep]
                    if not args.keep and not args.atomic:
                        os.remove(new)
                else:
                    reps = []
                    for chain in structure[0].get_chains():
                        new = f'{pdb[:-4]}_{chain.id}.pdb'
                        io.set_structure(structure[0][chain.id])
                        io.save(new, select=AtomSelect())
                        pdb_rep = create_representation(new, args)[0]
                        pdb_rep.chain = chain.id
                        reps.append(pdb_rep)
                        if not args.keep and not args.atomic:
                            os.remove(new)
            else:
                model = structure[0]
                new_chain = PDB.Chain.Chain('A')
                res_num = 1
                for chain in model.get_chains():
                    for residue in chain.get_residues():
                        new_residue = residue.copy()
                        new_residue.id = (' ', res_num, ' ')
                        new_chain.add(new_residue)
                        res_num += 1
                for chain in list(model.get_chains()):
                    model.detach_child(chain.id)
                model.add(new_chain)
                new = f'{pdb[:-4]}_combined.pdb'
                io.set_structure(structure)
                io.save(new, select=AtomSelect())
                pdb_rep = create_representation(new, args)[0]
                reps = [pdb_rep]
                if not args.keep and not args.atomic:
                    os.remove(new)
    else:
        reps = [pdb]

    if args.time:
        t1 = perf_counter()
        print(f'PDB -> NUMPY: {t1-t0} s')

    return reps

def predict(rep, models, avg, std, args, device):
    """
    Make prediction.

    Parameters
    ----------
    rep : NumpyRep or str
        Numpy representation of PDB structure or path to .npz file.
    models : dict
        Dictionary containing all models needed to make the specified prediction.
    avg : numpy.ndarray or float or tuple
        Average of the training dataset. (needed because predictions are normalized)
    std : numpy.ndarray or float or tuple
        Standard deviation of the training dataset. (needed because predictions are normalized)
    args : argparse.Namespace
        The command line arguments parsed by `argparse.ArgumentParser().parse_args()`.
    device : torch.device
        Specifies the device on which computations will be performed. ("cuda" -> GPU / "cpu" -> CPU)

    Returns
    -------
    None
    """
    if args.time:
        t0 = perf_counter()

    if not args.numpy:
        # Convert Numpy data to Torch tensor
        if not args.atomic:
            try:
                pos = torch.tensor(rep.x, dtype=torch.float).to(device)
                a = torch.tensor(rep.get_aas(), dtype=torch.long).to(device)
                cc = torch.tensor(rep.get_cc(), dtype=torch.float).reshape(-1, 1).to(device)
                dh = torch.tensor(rep.get_dh(), dtype=torch.float).to(device)
            except Exception as e:
                print(f'Error with {rep.pdb}. Check PDB for missing atoms.')
                return
    else:
        data = np.load(rep)
        if not args.atomic:
            # Load data directly from .npz file
            pos = torch.tensor(data['x'], dtype=torch.float).to(device)
            a = torch.tensor(data['a'], dtype=torch.long).to(device)
            cc = torch.tensor(data['cc'], dtype=torch.float).reshape(-1, 1).to(device)
            dh = torch.tensor(data['dh'], dtype=torch.float).to(device)
            if args.pka:
                resid = torch.tensor(data['resid']).to(device)
        else:
            pos = torch.tensor(data['x'], dtype=torch.float).to(device)
            a = torch.tensor(data['a'], dtype=torch.long).to(device)
            atoms = torch.tensor(data['atoms'], dtype=torch.long).to(device)
            charge = torch.tensor(data['charge'], dtype=torch.float).to(device)
            resid_atomic = torch.tensor(data['resid_atomic']).to(device)
            resid_ca = torch.tensor(data['resid_ca']).to(device)
    if args.time:
        t1 = perf_counter()
        print(f'NUMPY -> TORCH: {t1-t0} s')

    if args.pka:
        
        # Define standard pKa values
        standard = {
            'GLU': 4.07,
            'LYS': 10.54,
            'CYS': 8.37,
            'HIS': 6.04,
            'ASP': 3.90,
            'TYR': 10.46,
        }

        if not args.numpy:
            
            prot = rep.traj.top

            resids = []

            if args.atomic:
                pqr = '.'.join([frag for frag in rep.pdb.split('.')[:-1] + ['pqr']])
               
                if not os.path.isfile(pqr):
                    if args.time:
                        t0 = perf_counter()
                    
                    sub.run(
                        f'pdb2pqr30 --ff AMBER {rep.pdb} {pqr}', shell=True,
                        stderr=sub.DEVNULL, stdout=sub.DEVNULL
                    )

                    if args.time:
                        t1 = perf_counter()
                        print(f'PDB2PQR: {t1-t0} s')

                pqr_traj = md.load_pdb(pqr)

            for r in prot.residues:
                if (r_3 := r.__repr__()[:3]) in standard:
                    resSeq = r.resSeq

                    if not args.atomic:
                        idx = torch.tensor(rep.resSeq_to_resid(resSeq, mask=True))

                        if args.time:
                            t0 = perf_counter()
                        with torch.no_grad():
                            pred = models['gnn'](pos, 
                                                 a, 
                                                 cc, 
                                                 dh, 
                                                 resid=idx)

                        if args.time:
                            t1 = perf_counter()
                            print(f'FORWARD PASS: {t1-t0} s')
                    else:
                        if args.time:
                            t0 = perf_counter()
                        atomic_rep = NumpyRep_atomic(pqr,resSeq,traj=pqr_traj) 
                        if args.time:
                            t1 = perf_counter()
                            print(f'CREATE ATOMIC REPRESENTATION: {t1-t0} s')

                        if args.time:
                            t0 = perf_counter()
                        with torch.no_grad():
                            pred = models['gnn'](torch.tensor(atomic_rep.x, dtype=torch.float).to(device),
                                                 torch.tensor(atomic_rep.a, dtype=torch.long).to(device),
                                                 torch.tensor(atomic_rep.atoms, dtype=torch.long).to(device),
                                                 torch.tensor(atomic_rep.charge, dtype=torch.float).to(device),
                                                 resid_ca=torch.tensor(atomic_rep.resid_ca).to(device),
                                                 resid_atomic=torch.tensor(atomic_rep.resid_atomic).to(device))

                        if args.time:
                            t1 = perf_counter()
                            print(f'FORWARD PASS: {t1-t0} s')
                    pred = pred.cpu().detach().numpy()[0][0] * std + avg

                    if not args.shift:
                        pred += standard[r_3]

                    print_result('pka', pred, rep.pdb, resid=resSeq, aa=r_3, chain=rep.chain)
        else:
            with torch.no_grad():
                if not args.atomic: # Making a pKa prediction on an NPZ file with GSnet
                    pred = models['gnn'](pos, 
                                         a, 
                                         cc, 
                                         dh, 
                                         resid=resid)
                else: # Making a pKa prediction on an NPZ file with a-GSnet
                    pred = models['gnn'](pos, 
                                         a, 
                                         atoms, 
                                         charge, 
                                         resid_ca=resid_ca, 
                                         resid_atomic=resid_atomic)

                pred = pred.cpu().detach().numpy()[0][0] * std + avg
            
            print_result('pka', pred, rep)

    elif args.sasa:
        with torch.no_grad():
            h = models['gnn'](pos, a, cc, dh)
            pred = models['6'](h).cpu().numpy()[0] * std[0] + avg[0]
            sasa = models['sasa'](h).cpu().numpy()[0] * std[1] + avg[1]
            pred = np.append(pred, sasa)

        print_result('sasa', pred, rep.pdb, chain=rep.chain)

    else:
        with torch.no_grad():
            if args.time:
                t0 = perf_counter()
            pred = models['gnn'](pos, a, cc, dh)[0].cpu().numpy() * std + avg
            if args.time:
                t1 = perf_counter()
                print(f'FORWARD PASS: {t1-t0} s')

        print_result('default', pred, rep.pdb, chain=rep.chain)


def parse_arguments():
    """
    Parse command line arguments.
    """
    return parse_args()


def select_device(args):
    """
    Select the computation device.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments parsed by `argparse.ArgumentParser().parse_args()`.

    Returns
    -------
    torch.device
        The device to use for computation.
    """
    if args.gpu:
        assert torch.cuda.is_available(), 'You selected --gpu but no GPUs were detected on your system.'
    return torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')


def load_normalization_parameters(args, model_dir):
    """
    Load normalization parameters.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments parsed by `argparse.ArgumentParser().parse_args()`.
    model_dir : str
        Directory containing the models.

    Returns
    -------
    tuple
        Average and standard deviation for normalization.
    """
    normalization_params = np.load(f'{model_dir}/normalization.npz')

    if args.pka:
        #print(normalization_params['pka'])
        if args.atomic:
            return normalization_params['pka_a']
        else:
            return normalization_params['pka_r']
    elif args.sasa:
        print('ΔG [kJ/mol]   RG [Å]        RH [Å]        DT [nm^2/ns]  DR [ns^-1]    V [nm^3]      SASA [nm^2]   FILE')
        avg = [normalization_params['default'][0], normalization_params['sasa'][0]]
        std = [normalization_params['default'][1], normalization_params['sasa'][1]]
    else:
        print('ΔG [kJ/mol]   RG [Å]        RH [Å]        DT [nm^2/ns]  DR [ns^-1]    V [nm^3]      FILE')
        return normalization_params['default']
    return avg, std


def create_representations(args):
    """
    Create representations for each PDB file.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments parsed by `argparse.ArgumentParser().parse_args()`.

    Returns
    -------
    list
        List of representations for each PDB file.
    """
    representations = []
    for pdb in args.pdbs:
        representations += create_representation(pdb, args)
    return representations


def make_predictions(representations, models, avg, std, args, device):
    """
    Make predictions for each representation.

    Parameters
    ----------
    representations : list
        List of representations for each PDB file.
    models : dict
        Dictionary containing all models needed to make specified prediction.
    avg : numpy.ndarray or float or tuple
        Average of ing dataset.
    std : numpy.ndarray or float or tuple
        Standard deviation of ing dataset.
    args : argparse.Namespace
        The command line arguments parsed by `argparse.ArgumentParser().parse_args()`.
    device : torch.device
        The device to use for computation.
    """
    for rep in representations:
        try:
            predict(rep, models, avg, std, args, device)
        except KeyError as e:
            raise e
            sys.stderr.write(f'Error with {rep}: {e}\n')
        except RuntimeError as e:
            raise e
            sys.stderr.write(f'Error with {rep}: {e}\n')


def main():
    """
    Main function to parse arguments, load models, and make predictions on given PDB files.
    """
    model_dir = '../models'  # Directory containing the models

    args = parse_arguments()  # Parse command line arguments
    device = select_device(args)  # Select device for computation

    models = load_models(args, device)  # Load models based on arguments and device

    avg, std = load_normalization_parameters(args, model_dir)  # Load normalization parameters

    representations = create_representations(args)  # Create representations for each PDB file

    make_predictions(representations, models, avg, std, args, device)  # Make predictions for each representation


if __name__ == '__main__':
    main()
