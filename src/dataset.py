#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Classes for processing PDBs and generating datasets
#
# ------------------------------------------------------------------------------
# Citation Information
# ------------------------------------------------------------------------------
# Authors: Spencer Wozniak, Giacomo Janson, Michael Feig
# Emails: spencerwozniak1@gmail.com, mfeiglab@gmail.com
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
import matplotlib.pyplot as plt
import random
from scipy.stats import pearsonr
from time import strftime, localtime
from time import perf_counter as t_
import warnings
warnings.filterwarnings('ignore')

## Torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch_scatter import scatter

## Torch Geometric
import torch_geometric
from torch_geometric.data import Data, Dataset, download_url, extract_zip
from torch_geometric.data.makedirs import makedirs
from torch_geometric.nn import MessagePassing, radius_graph

## MDTraj
import mdtraj as md
from mdtraj import load_pdb
from mdtraj import compute_center_of_mass as calc_cofm

class NumpyRep:
    """
    A class used to represent protein structure.

    Attributes
    ----------
    pdb : str
        Path to the PDB file
    traj : mdtraj.Trajectory
        Allows calculations via mdtraj.
    x : np.Ndarray
        XYZ coordinates of the atoms of the protein.
    chain : str
        The protein chain.
    idx_map : dict
        Maps resSeq to resid.
    """
    def __init__(self, pdb):
        """
        Parameters
        ----------
        pdb : str
            Path to PDB file
        """
        self.pdb = pdb
        self.traj = load_pdb(pdb)
        self.x = self.get_coords()
        self.chain = None
        self.idx_map = None

    def resSeq_to_resid(self, resSeq, mask=True):
        """
        Maps PDB residue number to index starting with zero.

        Parameters
        ----------
        resSeq : int
            PDB residue number
        mask : bool, optional
            Choose whether output is a mask or an int. (default output is a mask)
        """
        if self.idx_map is None: # Only need to create self.idx_map if not already created
            i = 0
            self.idx_map = {}
            for r in self.traj.top.residues:
                if not list(r.atoms_by_name('CA')):
                    continue
                else:
                    self.idx_map[r.resSeq] = i
                    i+=1
        
        if mask:
            resid = np.zeros(len(self.x))
            resid[self.idx_map[resSeq]] +=1
            resid = resid.astype(bool)
        else:
            resid = self.idx_map[resSeq]

        return resid

    def get_coords(self):
        """
        Gets coordinate array from self.traj (md.Trajectory object)
        """
        ca_indices = [i for i, atom in enumerate(self.traj.topology.atoms) if atom.name == 'CA']
        return self.traj.xyz[0][ca_indices, :] * 10 # nm -> Å 
    
    def get_cc(self):
        """
        Calculates distances between alpha carbons and center of mass for all residues.
        """
        cofm = calc_cofm(self.traj)[0] * 10 # nm -> Å 
        x_minus_cofm = self.x - cofm[np.newaxis, :]
        return np.sqrt(np.einsum('ij,ij->i', x_minus_cofm, x_minus_cofm))
    
    def get_aas(self):
        """
        Gets integer representation of amino acid type for all residues.
        """

        # Histidine can have multiple labels (Diff ionization)
        h = ['HSD','HSE','HSP']

        # Arranged by keyboard layout
        AAs = ['GLN','TRP','GLU','ARG','THR','TYR',     'ILE',      'PRO',
               'ALA','SER','ASP','PHE','GLY','HIS',     'LYS','LEU',
                           'CYS','VAL',      'ASN','MET']
        
        aa_map = {aa: i for i, aa in enumerate(AAs)}
        aa_map['HIS'] = aa_map['HIS'] if 'HIS' in aa_map else aa_map['HSD']

        return np.array([aa_map.get(r.name, aa_map['HIS']) for r in self.traj.topology.residues])
   
    def get_dh(self):
        """
        Gets dihedral features (sine, cosine, mask) for all residues.
        """
        a2r = {}
        i = 0
        for r in self.traj.topology.residues:
            if not list(r.atoms_by_name('CA')):
                continue
            else:
                for a in r.atoms:
                    a2r[a.index] = i
                i += 1
        
        # Psi
        psis = np.array(list(md.compute_psi(self.traj)[1][0]) + [-2*np.pi]) 
        psis_mask = np.ones(len(psis))
        psis_mask[len(psis) - 1] = 0
        psis_sin = np.sin(psis)
        psis_sin[len(psis) - 1] = 0
        psis_cos = np.cos(psis)
        psis_cos[len(psis) - 1] = 0

        # Phi
        phis = np.array([-2*np.pi] + list(md.compute_phi(self.traj)[1][0]))
        phis_mask = np.ones(len(phis))
        phis_mask[0] = 0
        phis_sin = np.sin(phis)
        phis_sin[0] = 0
        phis_cos = np.cos(phis)
        phis_cos[0] = 0

        # Chi1
        chi1_data = md.compute_chi1(self.traj)
        chi1s = np.zeros(len(psis))
        chi1s_mask = np.ones(len(psis))
        chi1s += 10

        for i, chi in enumerate(chi1_data[1][0]):
            chi1s[a2r[chi1_data[0][i][0]]] = chi

        chi1s_sin = np.sin(chi1s)
        chi1s_cos = np.cos(chi1s)

        for i, chi in enumerate(chi1s):
            if chi == 10:
                chi1s_mask[i] = 0
                chi1s_sin[i] = 0
                chi1s_cos[i] = 0

        # Chi2
        chi2_data = md.compute_chi2(self.traj)
        chi2s = np.zeros(len(psis))
        chi2s_mask = np.ones(len(psis))
        chi2s += 10

        for i, chi in enumerate(chi2_data[1][0]):
            chi2s[a2r[chi2_data[0][i][0]]] = chi

        chi2s_sin = np.sin(chi1s)
        chi2s_cos = np.cos(chi1s)

        for i, chi in enumerate(chi2s):
            if chi == 10:
                chi2s_mask[i] = 0
                chi2s_sin[i] = 0
                chi2s_cos[i] = 0

        # Chi3
        chi3_data = md.compute_chi3(self.traj)
        chi3s = np.zeros(len(psis))
        chi3s_mask = np.ones(len(psis))
        chi3s += 10

        for i, chi in enumerate(chi3_data[1][0]):
            chi3s[a2r[chi3_data[0][i][0]]] = chi

        chi3s_sin = np.sin(chi3s)
        chi3s_cos = np.cos(chi3s)

        for i, chi in enumerate(chi3s):
            if chi == 10:
                chi3s_mask[i] = 0
                chi3s_sin[i] = 0
                chi3s_cos[i] = 0

        return np.array([psis_mask,
                         psis_sin,
                         psis_cos,
                         phis_mask,
                         phis_sin,
                         phis_cos,
                         chi1s_mask,
                         chi1s_sin,
                         chi1s_cos,
                         chi2s_mask,
                         chi2s_sin,
                         chi2s_cos,
                         chi3s_mask,
                         chi3s_sin,
                         chi3s_cos]).transpose()


class NumpyRep_atomic:
    """
    A class used to represent protein structure.

    Attributes
    ----------
    pdb : str
        Path to the PDB file
    resSeq : int
            Number of residue in PDB file
    cutoff : float
        Cutoff around alpha-carbon[resid] for considering atoms (in angstroms)
    traj : mdtraj.Trajectory
        Allows calculations via mdtraj.
    idx_map : dict
        Maps resSeq to mdtraj resid.
    resid_atomic : np.Ndarray
        Mask of atoms of interest in protein
    resid_ca : np.Ndarray
        Mask of alpha-carbons for atoms in protein
    cutoff_atoms : np.Ndarray
        Mask
    x : np.Ndarray
        XYZ coordinates of the atoms within the cutoff.
    a : np.Ndarray
        Amino acid encoding of atoms within the cutoff.
    atoms : np.Ndarray
        Atom encoding of atoms within the cutoff.
    charge : np.Ndarray
        Charges (calculated via pdb2pqr) of atoms within the cutoff.
    traj : md.Trajectory
        Allows passage of an MDTraj Trajectory object to accelerate computation.
    """
    def __init__(self, pdb, resSeq, cutoff=10.0, traj=None):
        """
        Parameters
        ----------
        pdb : str
            Path to PDB file
        resSeq : int
            Number of residue in PDB file
        cutoff : float, optional
            Cutoff for generating radius graph [in angstroms]
        """
        self.pdb = pdb
        self.resSeq = resSeq
        self.cutoff = cutoff
        
        if traj is None:
            self.traj = load_pdb(pdb)
        else:
            self.traj = traj

        self.x = self.get_coords()
        
        self.idx_map = None
        self.resid_atomic, self.resid_ca, self.cutoff_atoms = self.get_masks()
        
        self.x = self.x[self.cutoff_atoms]
        self.a = self.get_aas()[self.cutoff_atoms]
        self.atoms = self.get_atoms()[self.cutoff_atoms]
        self.charge = self.get_charge()[self.cutoff_atoms]

    def get_coords(self):
        """
        Gets coordinate array of all atoms in protein from self.traj (md.Trajectory object)
        """
        return self.traj.xyz[0] * 10 # nm -> Å 
    
    def get_masks(self, selection='subset'):
        """
        Generates masks.
        """
        selection = selection.lower()
        assert selection in {'all','subset'}, "Specify whether the masks are for 'all' atoms or the 'subset' defined by the cutoff"
        if self.idx_map is None: # Only need to create self.idx_map if not already created
            i = 0
            self.idx_map = {}
            for r in self.traj.top.residues:
                for index, atom in enumerate(r.atoms):
                    if "-CA" in str(atom):
                        ca_index = i+index
                        break
                self.idx_map[r.resSeq] = (i,i+r.n_atoms,ca_index)
                i+=r.n_atoms

        resid_atomic = np.zeros(len(self.x))
        resid_atomic[self.idx_map[self.resSeq][0]:self.idx_map[self.resSeq][1]] += 1
        
        resid_ca = np.zeros(len(self.x))
        resid_ca[self.idx_map[self.resSeq][2]] += 1
        
        resid_atomic, resid_ca = resid_atomic.astype(bool), resid_ca.astype(bool)

        x = self.x[resid_ca]
        x_minus_x = self.x - x
        dists = np.sqrt(np.einsum('ij,ij->i', x_minus_x, x_minus_x))
        cutoff_atoms = dists < self.cutoff
        cutoff_atoms = cutoff_atoms.astype(bool)
        
        if selection == 'subset':
            resid_atomic = resid_atomic[cutoff_atoms]
            resid_ca = resid_ca[cutoff_atoms]
        
        return resid_atomic, resid_ca, cutoff_atoms
    
    def get_aas(self):
        """
        Gets integer representation of amino acid type for all atoms.
        """

        # Histidine can have multiple labels (Diff ionization)
        h = ['HSD','HSE','HSP']

        # Arranged by keyboard layout
        AAs = ['GLN','TRP','GLU','ARG','THR','TYR',     'ILE',      'PRO',
               'ALA','SER','ASP','PHE','GLY','HIS',     'LYS','LEU',
                           'CYS','VAL',      'ASN','MET']
        
        aa_map = {aa: i for i, aa in enumerate(AAs)}
        aa_map['HIS'] = aa_map['HIS'] if 'HIS' in aa_map else aa_map['HSD']

        return np.array(
            [aa_map.get(a.residue.name, aa_map['HIS']) for a in self.traj.topology.atoms]
        )
    
    def get_atoms(self):
        """
        Gets integer representation of atom type for all atoms.
        """
        # Arranged in arbitrary order
        atoms = ['H','C','N','O','S']
        atom_map = {atom: i for i, atom in enumerate(atoms)}
        
        return np.array(
            [atom_map.get(a.name[0]) for a in self.traj.topology.atoms]
       )
        
    def get_charge(self):
        """
        Gets charge for all atoms (calculated via pdb2pqr).
        """
        with open(self.pdb, 'r') as f:
            return np.array(
                [float(line[55:63]) for line in f.readlines() if line.startswith('ATOM')]
            )


class ProteinDataset(Dataset):
    """
    Custom dataset used for training the GNN with various configurations.

    This dataset loads .npz files containing protein data, normalizes the data,
    and creates graph data objects for GNN training. It supports different configurations
    for data loading and preprocessing.

    Args:
        root (str): Path to the directory containing .npz files.
        avg (float, optional): Average value for normalization. Defaults to None.
        std (float, optional): Standard deviation for normalization. Defaults to None.
        use_dh (bool, optional): Whether to use DH feature. Defaults to False.
        use_cc (bool, optional): Whether to use CC feature. Defaults to False.
        use_res (bool, optional): Whether to use residue feature. Defaults to False.
        use_mask (bool, optional): Whether to use mask feature. Defaults to False.
        use_weight (bool, optional): Whether to use weight feature. Defaults to False.
        normalize (bool, optional): Whether to normalize the data. Defaults to True.
        label0 (bool, optional): Whether to use the first label value only. Defaults to False.
        check_files (bool, optional): Whether to check for valid .npz files. Defaults to False.
        skip_bad_files (bool, optional): Whether to skip bad files during the checking process. Defaults to False.
        dont_use (int, optional): Index of the feature to exclude from the data. Defaults to None.
        add_to_mask (bool, optional): Whether to add an additional value to the mask. Defaults to False.
        dG_index (bool, optional): Whether to use dG index feature. Defaults to False.
        use_cofactors (bool, optional): Whether to use cofactors feature. Defaults to False.
        nmr (str, optional): NMR configuration. You can choose "random" or "all" structures of an ensemble Defaults to None.
    """
    def __init__(self, root, avg=None, std=None, use_dh=False, use_cc=False, use_res=False, use_mask=False, use_weight=False, normalize=True, label0=False, check_files=False, skip_bad_files=False, dont_use=None, add_to_mask=False, dG_index=False, use_cofactors=False, nmr=None):
        super(ProteinDataset, self).__init__()
        self.root = root
        self.avg = avg
        self.std = std
        self.use_dh = use_dh
        self.use_cc = use_cc
        self.use_res = use_res
        self.use_mask = use_mask
        self.use_weight = use_weight
        self.normalize = normalize
        self.label0 = label0
        self.dont_use = dont_use
        self.add_to_mask = add_to_mask
        self.dG_index = dG_index
        self.use_cofactors = use_cofactors
        self.nmr = nmr
        self.cache = {}

        assert self.nmr in {None, 'random', 'all'}

        # Get .pdb files to .npz
        if check_files:
            assert self.nmr is None
            self._check_files(root, skip_bad_files)

        self.filelist = [f for f in os.listdir(self.root) if f.split('.')[-1] == 'npz'] if self.nmr is None else [f for f in os.listdir(self.root)]

        # Normalize the data if required
        if normalize:
            self.avg, self.std = self.normalize_()

    def _check_files(self, root, skip_bad_files):
        for f in os.listdir(root):
            ex = f.split('.')[-1]
            if ex == 'npz':
                continue
            elif ex == 'pdb':
                self.pdb_to_np(f'{root}/{f}', skip=skip_bad_files)
            else:
                raise TypeError('Invalid file type. (Must be PDB or NPZ)')

    def pdb_to_np(self, path, skip=False):
        try:
            rep = NumpyRep(path)
            np.savez(f'{path.strip(".pdb")}.npz',
                     x=rep.x,
                     a=rep.get_aas(),
                     dh=rep.get_dh(),
                     cc=rep.get_cc())
        except Exception as e:
            if skip:
                sys.stderr.write(f'Warning: {path} will not be added to dataset. (Bad PDB file)\n')
            else:
                raise e

    def len(self):
        return len(self.filelist)

    def normalize_(self):
        data = []
        for f in self.filelist:
            f_load = self._get_load_path(f)
            with np.load(f_load) as ary:
                data.append(ary['label'])
        data = np.asarray(data)
        avg = np.mean(data)
        std = np.std(data)
        return avg, std

    def _get_load_path(self, filename):
        if self.nmr is None:
            return f'{self.root}/{filename}'
        else:
            filename = [fn for fn in os.listdir(f'{self.root}/{filename}')][0]
            return f'{self.root}/{filename}/{filename}'

    def _load_data(self, f_load):
        data = np.load(f_load)
        if self.avg is not None and self.std is not None:
            z = (data['label'] - self.avg) / self.std
        else:
            z = data['label']
        m = self._get_mask(data, z)
        if self.label0:
            z = z[0]
        if self.dont_use is not None:
            z = np.concatenate([z[:self.dont_use], z[self.dont_use + 1:]])
            m = np.concatenate([m[:self.dont_use], m[self.dont_use + 1:]])
        return data, z, m

    def _get_mask(self, data, z):
        try:
            if not self.add_to_mask:
                return data['mask']
            else:
                return np.append(data['mask'], 1)
        except KeyError:
            return np.ones(len(z)) if isinstance(z, np.ndarray) else 1

    def get(self, inx):
        if inx in self.cache:
            return self.cache[inx]

        fn = self.filelist[inx]
        f_load_list = self._get_load_list(fn)
        graphs = [self._create_graph(f_load, inx) for f_load in f_load_list]

        if self.nmr != 'all':
            graph = graphs[0]
            self.cache[inx] = graph
            return graph
        else:
            self.cache[inx] = graphs
            return graphs

    def _get_load_list(self, fn):
        if self.nmr is None:
            return [f'{self.root}/{fn}']
        elif self.nmr == 'random':
            return [random.choice([f'{self.root}/{fn}/{model}' for model in os.listdir(f'{self.root}/{fn}')])]
        else:
            return [f'{self.root}/{fn}/{model}' for model in os.listdir(f'{self.root}/{fn}')]

    def _create_graph(self, f_load, inx):
        data, z, m = self._load_data(f_load)
        graph_data = {
            'idx': torch.tensor(inx, dtype=torch.long),
            'pos': torch.tensor(data['x'], dtype=torch.float32),
            'a': torch.tensor(data['a'], dtype=torch.long),
            'y': torch.tensor(z, dtype=torch.float32).reshape(1, -1)
        }
        if self.use_cc:
            graph_data['cc'] = torch.tensor(data['cc'], dtype=torch.float32).reshape(-1, 1)
        if self.use_dh:
            graph_data['dh'] = torch.tensor(data['dh'], dtype=torch.float32)
        if self.use_mask:
            graph_data['mask'] = torch.tensor(m, dtype=torch.long).reshape(1, -1)
        if self.use_res:
            graph_data['resid'] = torch.tensor(data['resid'])
            if self.use_cofactors:
                graph_data['cofactors'] = torch.tensor(data['cofactors'], dtype=torch.float32)
            if self.use_weight:
                graph_data['weight'] = torch.tensor(data['weight'], dtype=torch.float32)
        return Data(**graph_data)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        return self.get(idx)


class _ProteinDataset(ProteinDataset):
    """
    Modified ProteinDataset class for handling data with multi-dimensional target values.
    """
    def normalize(self):
        data = []
        for f in self.filelist:
            with np.load(f'{self.root}/{f}') as ary:
                data.append(ary['label'])
        data = np.asarray(data)
        avg = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return avg, std


class AtomicDataset(Dataset):
    """
    Custom dataset used for training the GNN with atomic features.

    This dataset loads .npz files containing atomic data, normalizes the data,
    and creates graph data objects for GNN training.

    Args:
        root (str): Path to the directory containing .npz files.
        avg (float, optional): Average value for normalization. Defaults to None.
        std (float, optional): Standard deviation for normalization. Defaults to None.
        normalize (bool, optional): Whether to normalize the data. Defaults to True.
        check_files (bool, optional): Whether to check for valid .npz files. Defaults to False.
        skip_bad_files (bool, optional): Whether to skip bad files during the checking process. Defaults to False.
    """
    def __init__(self, root, avg=None, std=None, normalize=True, cache_dir=None, check_files=False, skip_bad_files=False):
        super(AtomicDataset, self).__init__()
        self.root = root
        self.avg = avg
        self.std = std
        self.normalize = normalize
        self.filelist = [f for f in os.listdir(self.root) if f.split('.')[-1] == 'npz']

        # Check for valid files if required
        if check_files:
            self._check_files(root, skip_bad_files)

        # Normalize the data if required
        if normalize:
            self.avg, self.std = self.normalize_()
            
        # Caching
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _check_files(self, root, skip_bad_files):
        for f in os.listdir(root):
            ex = f.split('.')[-1]
            if ex == 'npz':
                continue
            elif ex == 'pdb':
                raise TypeError('PDB not currently supported. (Must be NPZ)')
            else:
                raise TypeError('Invalid file type. (Must be PDB or NPZ)')

    def pdb_to_np(self, path, resid, cutoff=10.0, skip=False):
        try:
            rep = NumpyRep_atomic(path, resid, cutoff)
            np.savez(f'{path.strip(".pdb")}.npz',
                     x=rep.x,
                     a=rep.a,
                     atoms=rep.atoms,
                     charge=rep.charge,
                     resid_atomic=rep.resid_atomic,
                     resid_ca=rep.resid_ca)
        except Exception as e:
            if skip:
                sys.stderr.write(f'Warning: {path} will not be added to dataset. (Bad PDB file)\n')
            else:
                raise e

    def len(self):
        return len(self.filelist)

    def normalize_(self):
        data = []
        for f in self.filelist:
            f_load = f'{self.root}/{f}'
            with np.load(f_load) as ary:
                data.append(ary['label'])

        data = np.asarray(data)
        avg = np.mean(data)
        std = np.std(data)
        return avg, std

    def get(self, inx):  # Same as '__getitem__'
        # Caching
        cache_path = None
        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, f'{inx}.pt')
            if os.path.exists(cache_path):
                try:
                    return torch.load(cache_path)
                except Exception as e:
                    sys.stderr.write(f"WARNING: Failed to load cache file {cache_path}: {e}\nThis file is likely corrupted. It will be removed.\n")
                    os.remove(cache_path)
                    
        fn = self.filelist[inx]

        f_load = f'{self.root}/{fn}'
        data = np.load(f_load)  # Load .npz file

        if self.avg is not None and self.std is not None:
            z = (data['label'] - self.avg) / self.std  # Z-score of label (normalized data)
        else:
            z = data['label']

        graph = Data(
            idx=torch.tensor(inx, dtype=torch.long),
            pos=torch.tensor(data['x'], dtype=torch.float32),
            a=torch.tensor(data['a'], dtype=torch.long),
            atom=torch.tensor(data['atoms'], dtype=torch.long),
            charge=torch.tensor(data['charge'], dtype=torch.float32).reshape(-1, 1),
            resid_atomic=torch.tensor(data['resid_atomic']),
            resid_ca=torch.tensor(data['resid_ca']),
            y=torch.tensor(z, dtype=torch.float32).reshape(1, -1)
        )
        
        if self.cache_dir is not None:
            torch.save(graph, cache_path)

        return graph

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        return self.get(idx)


class CombinedDataset(Dataset):
    """
    Custom dataset used for training the GNN with atomic and residue features.

    This dataset loads .npz files containing atomic and residue data, normalizes the data,
    and supports caching of processed graph data. It also handles loading of ESM and GearNet
    embeddings if specified.

    Args:
        root_atomic (str): Path to the directory containing atomic .npz files.
        root_res (str): Path to the directory containing residue .npz files.
        cache_dir (str, optional): Directory for caching processed graph data. Defaults to None.
        avg (float, optional): Average value for normalization. Defaults to None.
        std (float, optional): Standard deviation for normalization. Defaults to None.
        normalize (bool, optional): Whether to normalize the data. Defaults to True.
        esm_path (str, optional): Path to the directory containing ESM embeddings. Defaults to None.
        gearnet_path (str, optional): Path to the directory containing GearNet embeddings. Defaults to None.
        check_files (bool, optional): Whether to check for valid .npz files. Defaults to False.
        skip_bad_files (bool, optional): Whether to skip bad files during the checking process. Defaults to False.
    """
    def __init__(self, root_atomic, root_res, cache_dir=None, avg=None, std=None, normalize=True, esm_path=None, gearnet_path=None, check_files=False, skip_bad_files=False):
        super(CombinedDataset, self).__init__()
        self.root_atomic = root_atomic
        self.root_res = root_res
        self.avg = avg
        self.std = std
        self.normalize = normalize
        self.esm_path = esm_path
        self.gearnet_path = gearnet_path
        self.filelist = [f for f in os.listdir(self.root_atomic) if f.split('.')[-1] == 'npz']

        # Caching
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.bad_indices = []

        # Check for valid files if required
        if check_files:
            self._check_files(root_atomic, root_res, skip_bad_files)

        # Normalize the data if required
        if normalize:
            self.avg, self.std = self.normalize_()

    def _check_files(self, root_atomic, root_res, skip_bad_files):
        for ls in [os.listdir(root_atomic), os.listdir(root_res)]:
            for f in ls:
                ex = f.split('.')[-1]
                if ex == 'npz':
                    continue
                elif ex == 'pdb':
                    raise TypeError('PDB not currently supported. (Must be NPZ)')
                else:
                    raise TypeError('Invalid file type. (Must be PDB or NPZ)')

    def pdb_to_np(self, path, resid, cutoff=10.0, skip=False):
        try:
            rep = NumpyRep_atomic(path, resid, cutoff)
            np.savez(f'{path.strip(".pdb")}.npz',
                     x=rep.x,
                     a=rep.a,
                     atoms=rep.atoms,
                     charge=rep.charge,
                     resid_atomic=rep.resid_atomic,
                     resid_ca=rep.resid_ca)
        except Exception as e:
            if skip:
                sys.stderr.write(f'Warning: {path} will not be added to dataset. (Bad PDB file)\n')
            else:
                raise e

    def len(self):
        return len(self.filelist)

    def normalize_(self):
        data = []
        for f in self.filelist:
            f_load = f'{self.root_atomic}/{f}'
            with np.load(f_load) as ary:
                data.append(ary['label'])

        data = np.asarray(data)
        avg = np.mean(data)
        std = np.std(data)
        return avg, std

    def get(self, inx: int):
        # Caching
        cache_path = None
        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, f'{inx}.pt')
            if os.path.exists(cache_path):
                try:
                    return torch.load(cache_path)
                except Exception as e:
                    sys.stderr.write(f"WARNING: Failed to load cache file {cache_path}: {e}\nThis file is likely corrupted. It will be removed.\n")
                    os.remove(cache_path)

        if inx in self.bad_indices:
            return "INVALID"

        fn = self.filelist[inx]

        f_load = f'{self.root_atomic}/{fn}'
        data = np.load(f_load)  # Load .npz file

        if self.avg is not None and self.std is not None:
            z = (data['label'] - self.avg) / self.std  # Z-score of label (normalized data)
        else:
            z = data['label']

        f_load_res = f'{self.root_res}/{fn}'
        data_res = np.load(f_load_res)  # Load .npz file

        esm_fn = f'{"_".join(fn.split("_")[:-1])}.pt'

        if self.esm_path is None:
            graph_data = self._create_graph_data(inx, data, data_res, z)
        elif self.gearnet_path is None:
            try:
                esm_embedding = torch.load(f'{self.esm_path}/{esm_fn}')[data_res['resid']]
            except Exception:
                self.bad_indices.append(inx)
                return "INVALID"
            graph_data = self._create_graph_data(inx, data, data_res, z, esm_embedding=esm_embedding)
        else:
            try:
                esm_embedding = torch.load(f'{self.esm_path}/{esm_fn}')[data_res['resid']]
                gearnet_embedding = torch.load(f'{self.gearnet_path}/{esm_fn}')[data_res['resid']]
            except Exception:
                self.bad_indices.append(inx)
                return "INVALID"
            graph_data = self._create_graph_data(inx, data, data_res, z, esm_embedding=esm_embedding, gearnet_embedding=gearnet_embedding)

        if self.cache_dir is not None:
            torch.save(graph_data, cache_path)

        return graph_data

    def _create_graph_data(self, inx, data, data_res, z, esm_embedding=None, gearnet_embedding=None):
        graph_data = Data(
            idx=torch.tensor(inx, dtype=torch.long),
            pos_atomic=torch.tensor(data['x'], dtype=torch.float32),
            pos_res=torch.tensor(data_res['x'], dtype=torch.float32),
            a_atomic=torch.tensor(data['a'], dtype=torch.long),
            a_res=torch.tensor(data_res['a'], dtype=torch.long),
            atom=torch.tensor(data['atoms'], dtype=torch.long),
            charge=torch.tensor(data['charge'], dtype=torch.float32).reshape(-1, 1),
            resid_atomic=torch.tensor(data['resid_atomic']),
            resid_ca=torch.tensor(data['resid_ca']),
            resid_res=torch.tensor(data_res['resid']),
            cc=torch.tensor(data_res['cc'], dtype=torch.float32).reshape(-1, 1),
            dh=torch.tensor(data_res['dh'], dtype=torch.float32),
            y=torch.tensor(z, dtype=torch.float32).reshape(1, -1),
        )
        if esm_embedding is not None:
            graph_data.esm_embedding = esm_embedding
        if gearnet_embedding is not None:
            graph_data.gearnet_embedding = gearnet_embedding
        return graph_data

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        return self.get(idx)


if __name__ == '__main__':
    test_data = CombinedDataset(root_atomic=f'/feig/s1/spencer/gnn/data/pka/processed/cluster/deviation/0130/phmd/atomic/test',
                               root_res=f'/feig/s1/spencer/gnn/data/pka/processed/cluster/deviation/0130/phmd/test/',
                                esm_path=f'/feig/s1/spencer/gnn/compare/ESM/data/embeddings/650M/new',
                                normalize=False,
                               )

    test_data
    test_data[0]
