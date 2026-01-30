#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate GSnet and/or aLCnet datasets from residue-level CSV files.

Input CSV format:
PDB,CHAIN,RES,RES_IDX,PKA

- Downloads PDBs from RCSB if missing
- Extracts a single chain
- Writes clean PDBs to disk
- Generates CSVs referencing final PDB paths
- Generates NPZ files for GSnet and/or aLCnet

No pandas. No sklearn.
"""

import argparse
import csv
import os
import sys
import urllib.request
from typing import List, Dict

import numpy as np
from Bio import PDB

from dataset import NumpyRep, NumpyRep_atomic


# ----------------------------
# Constants
# ----------------------------

RCSB_URL = "https://files.rcsb.org/download/{}.pdb"


STANDARD_AAS = {
    'GLN','TRP','GLU','ARG','THR','TYR','ILE','PRO','ALA','SER','ASP',
    'PHE','GLY','HIS','LYS','LEU','CYS','VAL','ASN','MET','HSD','HSE','HSP'
}


# ----------------------------
# PDB utilities
# ----------------------------

class AtomSelect(PDB.Select):
    """Same filtering philosophy as prediction script."""

    def accept_atom(self, atom):
        name = atom.get_fullname().strip()
        if name.startswith("HETATM") or name.startswith("ANISOU"):
            return False
        return True

    def accept_residue(self, residue):
        return residue.get_resname() in STANDARD_AAS


def download_pdb(pdb_id: str, out_path: str):
    url = RCSB_URL.format(pdb_id.upper())
    print(f"Downloading {pdb_id} from RCSB...")
    urllib.request.urlretrieve(url, out_path)


def extract_chain(
    pdb_in: str,
    pdb_out: str,
    chain_id: str
):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(None, pdb_in)

    if chain_id not in structure[0]:
        raise ValueError(f"Chain {chain_id} not found in {pdb_in}")

    io = PDB.PDBIO()
    io.set_structure(structure[0][chain_id])
    io.save(pdb_out, select=AtomSelect())


# ----------------------------
# CSV utilities
# ----------------------------

def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(path: str, header: List[str], rows: List[List]):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ----------------------------
# Dataset generation
# ----------------------------

def process_csv(
    csv_path: str,
    pdb_root: str,
    out_root: str,
    dataset_type: str
):
    name = os.path.splitext(os.path.basename(csv_path))[0]
    base_dir = os.path.join(out_root, name)

    pdb_dir = os.path.join(base_dir, "pdbs")
    npz_dir = os.path.join(base_dir, "npz")
    csv_dir = os.path.join(base_dir, "csv")

    for d in (pdb_dir, npz_dir, csv_dir):
        os.makedirs(d, exist_ok=True)

    rows = read_csv(csv_path)

    gsnet_rows = []
    alcnet_rows = []

    for idx, r in enumerate(rows):
        pdb_id = r["PDB"].upper()
        chain = r["CHAIN"]
        resid = int(r["RES_IDX"])
        pka = float(r["PKA"])

        raw_pdb = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        chain_pdb = os.path.join(pdb_dir, f"{pdb_id}_{chain}.pdb")

        if not os.path.isfile(raw_pdb):
            download_pdb(pdb_id, raw_pdb)

        if not os.path.isfile(chain_pdb):
            extract_chain(raw_pdb, chain_pdb, chain)

        # ---------------- GSnet ----------------
        if dataset_type in ("gsnet", "both"):
            rep = NumpyRep(chain_pdb)
            np.savez(
                os.path.join(npz_dir, f"gsnet_{idx}.npz"),
                label=pka,
                x=rep.x,
                a=rep.get_aas(),
                dh=rep.get_dh(),
                cc=rep.get_cc(),
            )
            gsnet_rows.append([chain_pdb, pka])

        # ---------------- aLCnet ----------------
        if dataset_type in ("alcnet", "both"):
            rep = NumpyRep_atomic(chain_pdb, resid)
            np.savez(
                os.path.join(npz_dir, f"alcnet_{idx}.npz"),
                label=pka,
                x=rep.x,
                a=rep.a,
                atoms=rep.atoms,
                charge=rep.charge,
                resid_atomic=rep.resid_atomic,
                resid_ca=rep.resid_ca,
            )
            alcnet_rows.append([chain_pdb, resid, pka])

    # ---------------- Write output CSVs ----------------

    if gsnet_rows:
        write_csv(
            os.path.join(csv_dir, "gsnet.csv"),
            ["PDB", "Target"],
            gsnet_rows
        )

    if alcnet_rows:
        write_csv(
            os.path.join(csv_dir, "alcnet.csv"),
            ["PDB", "Res", "Target"],
            alcnet_rows
        )


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate GSnet / aLCnet datasets from residue-level CSVs"
    )

    parser.add_argument(
        "--input_csv",
        nargs="+",
        required=True,
        help="One or more input CSV files"
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Root output directory"
    )

    parser.add_argument(
        "--dataset",
        choices=["gsnet", "alcnet", "both"],
        default="both"
    )

    args = parser.parse_args()

    for csv_path in args.input_csv:
        process_csv(
            csv_path=csv_path,
            pdb_root=None,
            out_root=args.outdir,
            dataset_type=args.dataset
        )


if __name__ == "__main__":
    main()

