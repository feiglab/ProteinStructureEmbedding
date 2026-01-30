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


# Amino acid index-to-3-letter mapping (must match dataset.py AAs order)
AA_INDEX_TO_CODE = [
    'GLN', 'TRP', 'GLU', 'ARG', 'THR', 'TYR', 'ILE', 'PRO',
    'ALA', 'SER', 'ASP', 'PHE', 'GLY', 'HIS', 'LYS', 'LEU',
    'CYS', 'VAL', 'ASN', 'MET'
]

# Standard pKa values for titratable residues (used for absolute pKa when not --shift)
STANDARD_PKA = {
    'GLU': 4.07,
    'LYS': 10.54,
    'CYS': 8.37,
    'HIS': 6.04,
    'ASP': 3.90,
    'TYR': 10.46,
}


def parse_resid_from_npz_path(path):
    """
    Try to get residue number from npz filename when convention is {pdb}_{chain}_{resid}.npz.

    Parameters
    ----------
    path : str
        Path to the .npz file (e.g. .../1A2P_C_93.npz).

    Returns
    -------
    int or None
        Residue number if parsed from the last underscore-separated segment, else None.
    """
    try:
        base = os.path.splitext(os.path.basename(path))[0]
        parts = base.split('_')
        if len(parts) >= 3:
            return int(parts[-1])
    except (ValueError, IndexError):
        pass
    return None


def parse_chain_from_npz_path(path):
    """
    Try to get chain ID from npz filename when convention is {pdb}_{chain}_{resid}.npz.

    Parameters
    ----------
    path : str
        Path to the .npz file (e.g. .../1A2P_C_93.npz).

    Returns
    -------
    str or None
        Chain ID (e.g. 'C', 'A') if parsed from the second-to-last segment, else None.
    """
    try:
        base = os.path.splitext(os.path.basename(path))[0]
        parts = base.split('_')
        if len(parts) >= 3:
            return parts[-2]
    except IndexError:
        pass
    return None


def get_numpy_pka_residue_info(data, atomic=False, npz_path=None):
    """
    Extract residue index (for display) and 3-letter amino acid code from npz data.

    Parameters
    ----------
    data : np.lib.npyio.NpzFile
        Loaded npz file (e.g. from np.load(path)).
    atomic : bool
        True for aLCnet (per-atom) npz, False for GSnet (per-residue) npz.
    npz_path : str, optional
        Path to the npz file; used to parse residue from filename (e.g. 1A2P_C_93.npz)
        when resSeq is not in data.

    Returns
    -------
    tuple of (resid_display, aa_code)
        resid_display : int or None
            Residue number for display: PDB resSeq if in data, else 1-based
            position from mask/atomic layout, or from npz_path if parseable.
        aa_code : str or None
            3-letter amino acid code (e.g. 'GLU'), or None if not determinable.
    """
    try:
        if atomic:
            # resid_ca marks the CA of the residue of interest; a is per-atom aa index
            resid_ca = np.asarray(data['resid_ca'])
            a = np.asarray(data['a'])
            ca_idx = np.flatnonzero(resid_ca)
            if ca_idx.size == 0:
                return None, None
            aa_idx = int(a[ca_idx[0]])
            # Prefer explicit resSeq; else try filename (e.g. 1A2P_C_93.npz); else 1
            if 'resSeq' in data:
                resid_display = int(data['resSeq'])
            elif npz_path is not None:
                resid_display = parse_resid_from_npz_path(npz_path) or 1
            else:
                resid_display = 1
        else:
            # resid can be a mask (length n_residues) or a scalar (0-based index or resSeq)
            r = np.asarray(data['resid'])
            a = np.asarray(data['a'])
            n_residues = len(a)
            if r.ndim == 0 or r.size == 1:
                # Scalar: 0-based index into residue list -> display as 1-based position
                res_idx = int(r.flat[0])
                resid_display = int(data['resSeq']) if 'resSeq' in data else (res_idx + 1)
                aa_idx = int(a[res_idx]) if 0 <= res_idx < n_residues else -1
            else:
                # Mask: 1 at titratable residue
                res_indices = np.flatnonzero(r)
                if res_indices.size == 0:
                    return None, None
                res_idx = int(res_indices[0])
                aa_idx = int(a[res_idx])
                resid_display = int(data['resSeq']) if 'resSeq' in data else (res_idx + 1)
        if 0 <= aa_idx < len(AA_INDEX_TO_CODE):
            aa_code = AA_INDEX_TO_CODE[aa_idx]
            return resid_display, aa_code
        return resid_display, None
    except (KeyError, IndexError, TypeError):
        return None, None


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
    parser.add_argument('--show-label', action='store_true',
                        help='With --numpy: include observed value (label) from npz in pKa output.')
    parser.add_argument('--state-dict', metavar='PATH', default=None,
                        help='Path to custom state dict (.pt) to load for the main model instead of the default.')
    parser.add_argument('pdbs', nargs='+', help='List of PDB files.')

    args = parser.parse_args()

    # Assertions to ensure valid argument combinations
    assert not (args.cpu and args.gpu), 'Cannot run on both CPU and GPU!'
    assert not ((args.chain is not None) and args.combine_chains), f'Cannot select chain {args.chain} and combine chains.'
    assert not (args.sasa and args.pka), 'Cannot calculate SASA when predicting pKa.'

    if args.show_label:
        assert args.numpy, "'--show-label' is only valid when using --numpy."

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


def _state_dict_from_file(path, device):
    """
    Load state dict from a .pt file. Accepts raw state dict or checkpoint dict with
    'state_dict' / 'model_state_dict' key.
    """
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch (<2.0) does not support weights_only
        obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and 'state_dict' in obj:
        return obj['state_dict']
    if isinstance(obj, dict) and 'model_state_dict' in obj:
        return obj['model_state_dict']
    return obj


def load_state_dict_loose(model, path, device, context_hint=''):
    """
    Load state dict into model with strict=False. Raise if no parameters were loaded.

    Parameters
    ----------
    model : torch.nn.Module
        Model to load into.
    path : str
        Path to .pt file (state dict or checkpoint).
    device : torch.device
        Map location for tensors.
    context_hint : str
        Short hint for error message (e.g. 'pKa GSnet', 'aLCnet').

    Raises
    ------
    ValueError
        If no model parameters were loaded (state dict incompatible with architecture).
    """
    state_dict = _state_dict_from_file(path, device)
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"File {path} does not contain a state dict (got {type(state_dict).__name__}). "
            "Expected a .pt file with a state dict or checkpoint dict."
        )
    result = model.load_state_dict(state_dict, strict=False)
    num_model_keys = len(model.state_dict())
    num_loaded = num_model_keys - len(result.missing_keys)
    if num_loaded == 0:
        raise ValueError(
            f"The state dict at {path} is incompatible with the selected architecture. "
            "No parameters were loaded. "
            f"Check that the state dict matches this model type ({context_hint or 'e.g. --atomic vs not'})."
        )


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
    use_custom_state_dict = getattr(args, 'state_dict', None) is not None

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

            if use_custom_state_dict:
                load_state_dict_loose(models['gnn'], args.state_dict, device, context_hint='pKa GSnet (non-atomic)')
            else:
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
            if use_custom_state_dict:
                load_state_dict_loose(models['gnn'], args.state_dict, device, context_hint='pKa aLCnet (--atomic)')
            else:
                state_dict = torch.load(f'{model_dir}/aLCnet_pKa.pt', map_location=device)
                models['gnn'].load_state_dict(state_dict, strict=False)
    elif args.sasa:
        models['gnn'] = Net(embedding_only=True)
        state_dict = torch.load(f'{model_dir}/GSnet_default.pt', map_location=device)
        if use_custom_state_dict:
            load_state_dict_loose(models['gnn'], args.state_dict, device, context_hint='SASA GNN')
        else:
            models['gnn'].load_state_dict(state_dict, strict=False)

        models['sasa'] = FC(in_channels=150, num_linear=4, out_channels=1, dropout=0.0)
        fc_elements = OrderedDict((f'fc{k[4:]}', v) for k, v in torch.load(f'{model_dir}/GSnet_SASA.pt', map_location=device).items() if k.startswith('fc_t'))
        models['sasa'].load_state_dict(fc_elements)

        models['6'] = FC(in_channels=150, num_linear=4, out_channels=6, dropout=0.0)
        fc_elements = OrderedDict((k, v) for k, v in state_dict.items() if k.startswith('fc.'))
        models['6'].load_state_dict(fc_elements)
    else:
        models['gnn'] = Net()
        if use_custom_state_dict:
            load_state_dict_loose(models['gnn'], args.state_dict, device, context_hint='default GSnet')
        else:
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


def get_label_from_npz(data):
    """
    Safely get observed value (label) from npz data if present.
    Labels in pKa npz files are stored as shifts (not absolute pKa).

    Parameters
    ----------
    data : np.lib.npyio.NpzFile
        Loaded npz file.

    Returns
    -------
    float or None
        Label value (shift), or None if missing or not convertible to float.
    """
    try:
        if 'label' not in data:
            return None
        return float(np.asarray(data['label']).flat[0])
    except (TypeError, ValueError, IndexError):
        return None


def observed_label_to_display(observed_shift, shift_mode, aa_code):
    """
    Convert observed label (stored as shift in npz) to the value to display,
    matching the predicted column: shift when --shift, else absolute pKa.

    Parameters
    ----------
    observed_shift : float or None
        Raw label from npz (pKa shift).
    shift_mode : bool
        True if user passed --shift (output is shifts).
    aa_code : str or None
        3-letter amino acid code (e.g. 'GLU'), used to add standard pKa when not shift_mode.

    Returns
    -------
    float or None
        Value to print in Observed column, or None to show "-".
    """
    if observed_shift is None:
        return None
    if shift_mode:
        return observed_shift
    if aa_code is not None and aa_code in STANDARD_PKA:
        return observed_shift + STANDARD_PKA[aa_code]
    return observed_shift


def _pka_header(show_label=False):
    """Return the pKa output table header string."""
    if show_label:
        return 'Predicted  Observed  AA    Res   Chain  File'
    return 'Predicted  AA    Res   Chain  File'


def print_result(calc_type, pred, pdb, chain=None, resid=None, aa=None, observed=None, show_label=False):
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
    observed : float, optional
        Observed value (e.g. from npz 'label'). Only used when calc_type=='pka' and show_label is True.
    show_label : bool, optional
        If True and calc_type=='pka', include observed column (use "-" when observed is None).
    """
    if calc_type == 'pka':
        chain_str = '-' if chain is None else chain
        if show_label:
            obs_str = observed if observed is not None else '-'
            print(pred, obs_str, aa, resid, chain_str, pdb)
        else:
            print(pred, aa, resid, chain_str, pdb)
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
                if (r_3 := r.__repr__()[:3]) in STANDARD_PKA:
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
                        pred += STANDARD_PKA[r_3]

                    print_result('pka', pred, rep.pdb, resid=resSeq, aa=r_3, chain=rep.chain,
                                 observed=None, show_label=args.show_label)
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

            # Only add standard pKa when not --shift (same as PDB path)
            resid_display, aa_code = get_numpy_pka_residue_info(
                data, atomic=args.atomic, npz_path=rep
            )
            if not args.shift and aa_code is not None and aa_code in STANDARD_PKA:
                pred += STANDARD_PKA[aa_code]
            observed_raw = get_label_from_npz(data) if args.show_label else None
            observed = observed_label_to_display(observed_raw, args.shift, aa_code)
            chain = parse_chain_from_npz_path(rep)
            print_result('pka', pred, rep, chain=chain, resid=resid_display, aa=aa_code,
                         observed=observed, show_label=args.show_label)

    elif args.sasa:
        with torch.no_grad():
            h = models['gnn'](pos, a, cc, dh)
            pred = models['6'](h).cpu().numpy()[0] * std[0] + avg[0]
            sasa = models['sasa'](h).cpu().numpy()[0] * std[1] + avg[1]
            pred = np.append(pred, sasa)

        file_path = rep if args.numpy else rep.pdb
        chain = None if args.numpy else rep.chain
        print_result('sasa', pred, file_path, chain=chain)

    else:
        with torch.no_grad():
            if args.time:
                t0 = perf_counter()
            pred = models['gnn'](pos, a, cc, dh)[0].cpu().numpy() * std + avg
            if args.time:
                t1 = perf_counter()
                print(f'FORWARD PASS: {t1-t0} s')

        file_path = rep if args.numpy else rep.pdb
        chain = None if args.numpy else rep.chain
        print_result('default', pred, file_path, chain=chain)


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
        print(_pka_header(show_label=args.show_label))
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
