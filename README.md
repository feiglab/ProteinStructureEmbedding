# Accurate Predictions of Molecular Properties of Proteins via Graph Neural Networks and Transfer Learning

## Overview

### Graph Neural Network-Based Prediction

This repository contains code and pre-trained weights for **[Graph Transformer Networks](https://pytorch-geometric.readthedocs.io/en/2.5.1/generated/torch_geometric.nn.conv.TransformerConv.html)** that are trained for the computational prediction of protein properties. **[GSnet](#gsnet)** is adept at predicting a variety of physicochemical properties from three-dimensional protein structures, while **[aLCnet](#alcnet)** was specifically trained for residue-specific pKa prediction. Moreover, the application of **[transfer learning](https://en.wikipedia.org/wiki/Transfer_learning)** allows these models to utilize previously learned representations (_i.e._, **embeddings**) in new prediction tasks, even with limited specific training data.

Properties that these models can predict include:

- Free energy of solvation ($\Delta G_{sol}$)
- Hydrodynamic radius ($R_h$)
- Translational diffusion coefficient ($D_t$)
- Rotational diffusion coefficient ($D_r$)
- Molecular volume ($V$)
- Radius of gyration ($R_g$)
- Solvent accessible surface area ($SASA$)
- $pK_a$ values

<details open><summary><b>Model Architecture</b></summary>

### GSnet

![GSNet architecture dark](https://github.com/user-attachments/assets/25b0a19e-9f4b-4868-a1dd-66e54a946831)

### aLCnet

![aLCnet architecture dark](https://github.com/user-attachments/assets/be72981f-f494-4bf0-a781-8a1de175171b)
</details>

<details><summary><b>Paper</b></summary>

For more information about this project, you can access the paper at the following link:

**[Accurate Predictions of Molecular Properties of Proteins via Graph Neural Networks and Transfer Learning](https://doi.org/10.1021/acs.jctc.4c01682)**

To cite our project in your work, use:

```bibtex
@article{doi:10.1021/acs.jctc.4c01682,
author = {Wozniak, Spencer and Janson, Giacomo and Feig, Michael},
title = {Accurate Predictions of Molecular Properties of Proteins via Graph Neural Networks and Transfer Learning},
journal = {Journal of Chemical Theory and Computation},
volume = {21},
number = {9},
pages = {4830-4845},
year = {2025},
doi = {10.1021/acs.jctc.4c01682},
note = {PMID: 40270304},
URL = {https://doi.org/10.1021/acs.jctc.4c01682},
eprint = {https://doi.org/10.1021/acs.jctc.4c01682}
}
```

</details>


<details open><summary><b>Table of Contents</b></summary>

- [Installation](#installation)
- [Making Predictions](#making-predictions)
- [Plotting predictions](#plotting-predictions)
- [Pretrained Models](#pretrained-models)
- [Generating Embeddings](#generating-embeddings)
- [Generating Datasets](#generating-datasets)
  - [GSnet Dataset](#gsnet-dataset)
  - [aLCnet Dataset](#alcnet-dataset)
- [Training a Model](#training-a-model)
- [How to reproduce data and figures from the paper](#how-to-reproduce-data-and-figures-from-the-paper)
- [Directory Structure](#directory-structure)
</details>

## Installation

Before you can run the models, you need to set up your environment:

1. **Clone the repository:**

```bash
git clone https://github.com/feiglab/ProteinStructureEmbedding.git
cd ProteinStructureEmbedding
```

2. **Set up the environment:**
   - If you use a virtual environment, set it up and activate it before installing the packages:

```bash
conda create -n gsnet python=3.9 pip
conda activate gsnet
```
    
3. **Install required packages:**
   - Use Python 3.9 or 3.10 on your system.
   - Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Making Predictions

The `predict.py` script predicts physicochemical properties from PDB files or from precomputed NPZ representations. Run from the `src/` directory (or with `src` on `PYTHONPATH`). You can pass multiple input files.

### Modes

- **Default** — Six global properties: ΔG, RG, RH, DT, DR, V.
- **`--sasa`** — Same six plus SASA.
- **`--pka`** — Residue-level pKa for titratable residues (ASP, CYS, GLU, HIS, LYS, TYR). Use **`--atomic`** for aLCnet (faster and more accurate); omit for GSnet. Use **`--shift`** to output pKa shift from the standard value instead of absolute pKa.

pKa output is tabular with a header. Columns: Predicted, (Observed if `--show-label`), AA, Res, Chain, File.

### Input: PDB vs NPZ

- **PDB** — One or more `.pdb` files. Use **`--clean`** to strip non-standard residues/atoms before prediction; **`--keep`** keeps the cleaned files. **`--chain`** restricts to one chain; **`--combine-chains`** merges chains into one structure.
- **`--numpy`** — One or more `.npz` files (e.g. from `generate_datasets.py`). Same prediction modes and outputs as PDB. Residue index and chain are taken from the NPZ when present, otherwise parsed from the filename when it follows `{pdb}_{chain}_{resid}.npz`.

### pKa with NPZ: observed values

With **`--numpy`** and **`--show-label`**, the script prints an **Observed** column. NPZ labels are stored as **shifts**; the script converts them to absolute pKa when `--shift` is not used (using the residue type). If a file has no `label` key, Observed is printed as `-`.

### Custom model weights

**`--state-dict PATH`** loads a custom state dict for the main model instead of the default `.pt` file. Loading is loose (`strict=False`); if no parameters match the selected architecture, the script exits with an error. Supports raw state dicts or checkpoint dicts with a `state_dict` / `model_state_dict` key.

### Device and other options

- **`--cpu`** / **`--gpu`** — Force device (default: GPU if available).
- **`--time`** — Print timing for loading and forward pass.
- **`--skip-bad-files`** — Skip inputs that fail to load instead of raising.

### Example commands

**Default (global properties) from PDB:**
```bash
python predict.py /path/to/file.pdb
```

**pKa (absolute) with aLCnet:**
```bash
python predict.py --pka --atomic /path/to/file.pdb
```

**pKa shifts from NPZ with observed labels and custom weights:**
```bash
python predict.py --pka --atomic --numpy --shift --show-label --state-dict ../models/tr_25_test.pt ../pKa-datasets/msu-test-data/*.npz
```

**Sample pKa output (with header and observed):**
```
Predicted  Observed  AA    Res   Chain  File
-0.99      -1.67     ASP   93    C      ../pKa-datasets/msu-test-data/1A2P_C_93.npz
...
```

For all options: `python predict.py -h`

## Plotting predictions

The `plot_predictions.py` script plots **predicted vs observed** from the text output of `predict.py`. It expects output that includes both a **Predicted** and an **Observed** column (e.g. pKa runs with **`--show-label`**). Run from the `src/` directory.

**Default behavior:** the script **saves** the figure to **`predictions.png`** in the current directory (it does not open a window). Use **`--show`** to display the figure in a GUI instead of saving. Use **`-o path`** or **`--save path`** to save to a different path.

### Input

- **From file:** one or more files containing `predict.py` stdout (e.g. after redirecting: `python predict.py ... > out.txt`).
- **From stdin (pipe):** use `-` as the input so you can pipe `predict.py` directly into the plot script.

### Piping predict.py into the plot script

Pipe the output of `predict.py` into `plot_predictions.py`. By default the figure is saved as `predictions.png`:

```bash
python predict.py --pka --atomic --numpy --shift --show-label ../pKa-datasets/msu-test-data/*.npz | python plot_predictions.py -
```

Use `-o` to save under a different name:

```bash
python predict.py --pka --atomic --numpy --shift --show-label ... | python plot_predictions.py - -o pka_plot.png
```

You can add other plot options before the output path:

```bash
python predict.py --pka --atomic --numpy --show-label ... | python plot_predictions.py - --title "pKa" --linreg -o pka.png
```

### Using a saved output file

Save `predict.py` output to a file, then pass that file to the plot script (figure is saved as `predictions.png` unless you pass `-o`):

```bash
python predict.py --pka --atomic --numpy --show-label ... > pka_out.txt
python plot_predictions.py pka_out.txt
```

To save under a specific path:

```bash
python plot_predictions.py pka_out.txt -o pka_plot.png
```

You can pass multiple files; their data are combined into one plot.

### Plot options (defaults in script)

- **`--hexbin`** — Use hexbin instead of scatter (default: scatter).
- **`--linreg`** — Add linear regression line (default: on).
- **`--mape`** — Use MAPE instead of Pearson r on the 1:1 line.
- **`--title`**, **`--xlabel`**, **`--ylabel`**, **`--units`** — Labels and title.
- **`--clean`** — Drop rows where reference (x) &lt; -700.
- **`--save` / `-o`** — Save figure to this path (default when not using `--show`: `predictions.png`).
- **`--show`** — Display the figure in a GUI instead of saving.
- **`--dpi`** — Resolution for saved figure (default 600).

For all options: `python plot_predictions.py -h`

## Pretrained Models

| Model Name           | Number of Parameters | Description                                                                                     |
|----------------------|----------------------|-------------------------------------------------------------------------------------------------|
| `GSnet_default.pt`    | 5,971,748       | The original GSnet model trained on the 6 physicochemical properties. |
| `GSnet_SASA.pt`       | 5,971,748       | GSnet fine-tuned for molecular SASA predictions.                  |
| `GSnet_pKa.pt`        | 11,210,392       | GSnet fine-tuned for residue-level SASA, then further trained to predict pKa values.              |
| `aLCnet_pKa.pt`       | 4,784,324       | aLCnet trained from scratch on PHMD549 data and fine-tuned for pKa prediction on experimental data. |


## Generating Embeddings

The `embed_GSnet.py` and `embed_aLCnet.py` scripts allow you to easily generate embeddings for all PDB files within a specified directory.

- Generated embeddings (via either method) will be saved as tensors of shape `[N,d]` where `N` is the number of residues in the protein and `d` is the embedding dimension.

### Steps to Generate Embeddings

1. **Gather PDB files:**
 Put PDB files containing only 1 chain that you would like embeddings for into a directory. Make sure the file extension for the files is `.pdb`.

2. **Run the script:**
 Navigate to the `src/` directory in your terminal. Use the following command(s) to generate embeddings:

#### GSnet embeddings

 ```bash
 python embed_GSnet.py --protein/--residue PDBPATH OUTPATH
 ```
 Replace `PDBPATH` with the directory containing your PDB files and `OUTPATH` with the directory where you want to save the embeddings.

- Use the `--protein` option to generate GSnet embeddings optimized for **whole protein predictions** (trained on 6 physicochemical properties).
- Use the `--residue` option to generate GSnet embeddings optimized for **residue-specific predictions** (fine-tuned on rSASA and pKa).
- In theory, either embedding method (`--protein` or `--residue`) may be useful in either context. It could be worthwhile to try both embeddings for the same task to determine which is more useful.

#### aLCnet embeddings

 ```bash
 python embed_aLCnet.py PDBPATH OUTPATH
 ```
 Replace `PDBPATH` with the directory containing your PDB files and `OUTPATH` with the directory where you want to save the embeddings.

- This will take longer than GSnet embeddings because separate graphs will be constructed for atoms around each residue, rather than for the whole protein.

### Notes

- The scripts utilize multiprocessing to expedite the embedding process. Ensure your system has adequate resources to handle multiple processes simultaneously.

## Generating Datasets

Datasets for GSnet and aLCnet can be created in two ways: (1) using the **`generate_datasets.py`** script from residue-level CSV files (recommended when you have PDB IDs and pKa or other residue-level targets), or (2) manually with the **`dataset.py`** classes `NumpyRep` / `NumpyRep_atomic` and then `ProteinDataset` / `AtomicDataset`. The script automates downloading, chain extraction, and NPZ generation; the manual approach is for custom pipelines or when your data is already in a different format.

### Using the generate_datasets.py script

The `src/generate_datasets.py` script builds GSnet and/or aLCnet datasets from residue-level CSV files. It downloads PDBs from the RCSB if missing, extracts a single chain, writes cleaned single-chain PDBs to disk, and generates NPZ files (and output CSVs) in the format expected by `ProteinDataset` and `AtomicDataset`.

**Input CSV format**

Each input CSV must have the following columns:

| Column   | Description                                      |
|----------|--------------------------------------------------|
| `PDB`    | PDB ID (e.g. `1abc`) or path to a local PDB file |
| `CHAIN`  | Chain ID to use (e.g. `A`)                       |
| `RES`    | Residue name (informational)                     |
| `RES_IDX`| Residue index (1-based) for the target residue   |
| `PKA`    | Target value (e.g. pKa) for that residue         |

**How the script works**

1. For each row in the CSV, the script resolves the PDB file: if `PDB` looks like a PDB ID (e.g. `1abc`), it downloads the file from the RCSB into the output tree if not already present; otherwise it uses the given path.
2. It extracts the requested chain and keeps only standard amino acids (same filtering as the prediction script), writing a cleaned single-chain PDB (e.g. `{PDB}_{CHAIN}.pdb`) into the output directory.
3. Using the cleaned PDB and the row’s `RES_IDX` and `PKA`, it builds the appropriate representations and saves NPZ files for GSnet and/or aLCnet (depending on `--dataset`).
4. It writes summary CSVs (`csv/gsnet.csv` and/or `csv/alcnet.csv`) that list the final PDB paths and target values, matching the layout expected by the manual workflow below.

**Command**

From the project root (or with `src` on `PYTHONPATH`):

```bash
python src/generate_datasets.py --input_csv /path/to/file1.csv [/path/to/file2.csv ...] --outdir /path/to/output [--dataset gsnet|alcnet|both]
```

- **`--input_csv`**: One or more input CSV files (residue-level, format above).
- **`--outdir`**: Root directory under which all outputs are written.
- **`--dataset`**: `gsnet`, `alcnet`, or `both` (default: `both`). Controls whether to generate GSnet NPZs, aLCnet NPZs, or both.

**Output layout**

For each input CSV file, the script creates a subdirectory under `--outdir` named after the CSV (without the `.csv` extension). Inside that subdirectory:

| Path       | Contents                                                                 |
|------------|--------------------------------------------------------------------------|
| `pdbs/`    | Downloaded and/or chain-extracted PDBs (e.g. `1abc.pdb`, `1abc_A.pdb`).  |
| `npz/`     | NPZ files: `gsnet_0.npz`, `gsnet_1.npz`, ... and/or `alcnet_0.npz`, `alcnet_1.npz`, ... (one per CSV row). |
| `csv/`     | Summary CSVs: `gsnet.csv` (columns `PDB`, `Target`) and/or `alcnet.csv` (columns `PDB`, `Res`, `Target`). |

The NPZ files have the same structure as in the manual workflow below. You can load them with `ProteinDataset` (for `npz/gsnet_*.npz`) or `AtomicDataset` (for `npz/alcnet_*.npz`) by passing the `npz` directory as `root` (e.g. `root='/path/to/output/my_dataset/npz'` for a dataset named `my_dataset`). Splitting into train/val/test is done by organizing or symlinking NPZ directories and then creating separate `ProteinDataset` / `AtomicDataset` instances for each split.

### GSnet Dataset

1. **Have paths to PDBs and target values stored in a CSV file (or similar):**

```csv
PDB,Target Value
/path/to/file1.pdb,4.10
/path/to/file2.pdb,6.21
/path/to/file3.pdb,7.94
...
```

2. **Generate NumPy representations of the data:**

```python
import numpy as np
import pandas as pd
from dataset import NumpyRep

outdir = '/path/to/output/dir'

df = pd.read_csv('/path/to/file.csv') # Read CSV file

# Iterate over datapoints in dataset (this can be expidited with multiprocessing)
for i, row in df.iterrows():
    rep = NumpyRep(row[0]) # Create a NumpyRep for PDB
    y = float(row[1])      # Extract target value

    # We want to generate NPZ files for each datapoint
    np.savez(
        f'{outdir}/{i}.npz', # Define output file path
        label = y,           # Define target value
        x = rep.x,           # Define Cartesian coordinates of residues
        a = rep.get_aas(),   # Define residue types
        dh = rep.get_dh(),   # Define dihedral information
        cc = rep.get_cc()    # Define alpha carbon to center of mass distance
    )
```

3. **Generate a PyTorch dataset:**

```python
import numpy as np
from dataset import ProteinDataset

dataset = ProteinDataset(
    root='/path/to/output/dir', # Path to directory containing NPZ files
    use_dh=True,                # Specify that dihedral info is used
    use_cc=True,                # Specify that ca-cofm distance is used
    normalize=True              # Normalize target values
)
```

### aLCnet Dataset

1. **Have paths to PDBs, residue indicies, and target values stored in a CSV file (or similar):**

```csv
PDB,Res,Target Value
/path/to/file1.pdb,24,4.10
/path/to/file2.pdb,54,6.21
/path/to/file3.pdb,91,7.94
...
```

2. **Generate NumPy representations of the data:**

```python
import numpy as np
import pandas as pd
from dataset import NumpyRep_atomic

outdir = '/path/to/output/dir'

df = pd.read_csv('/path/to/file.csv') # Read CSV file

# Iterate over datapoints in dataset (this can be expidited with multiprocessing)
for i, row in df.iterrows():
    rep = NumpyRep_atomic(row[0],row[1]) # Create a NumpyRep for residue in PDB
    y = float(row[2])                    # Extract target value

    # We want to generate NPZ files for each datapoint
    np.savez(
        f'{outdir}/{i}.npz',             # Define output file path
        label = y,                       # Define target value
        x = rep.x,                       # Define Cartesian coordinates of residues
        a = rep.a,                       # Define residue types
        atoms = rep.atoms,               # Define atom types
        charge = rep.charge,             # Define atom charges
        resid_atomic=rep.resid_atomic,   # Define residue atom indicies
        resid_ca=rep.resid_ca,           # Define alpha-carbon index
    )
```

3. **Generate a PyTorch dataset:**

```python
import numpy as np
from dataset import AtomicDataset

dataset = AtomicDataset(
    root='/path/to/output/dir', # Path to directory containing NPZ files
    normalize=True              # Normalize target values
)
```

### Notes

- You can split the NPZ data into multiple directories to have training, validation, test sets via any method you choose. You can then load multiple PyTorch datasets.

## Training a Model

Sample training scripts `train_GSnet.py` and `train_aLCnet.py` are provided for training GSnet and aLCnet, respectively.

To train a new model:

1. Make sure you have PyTorch datasets generated. See [Generating Datasets](#generating-datasets) for more info.
2. See the `train_GSnet.py` and `train_aLCnet.py` scripts for examples on how to train our models. Sample data for training both GSnet and aLCnet was provided for selected structures.

## How to reproduce data and figures from the paper

1. **Install the repo** — [Installation](#installation)
2. **Generate NPZ data sets** — [Generating Datasets](#generating-datasets)
3. **Make predictions** — [Making Predictions](#making-predictions)
4. **Plot predictions** — Run `plot_predictions.py` on the tabular output of `predict.py`. Use **`--show-label`** with `predict.py` so the output includes an Observed column; then pipe that output (optionally filtered, e.g. by residue type) into `plot_predictions.py`. By default the plot is saved to `predictions.png`; use **`-o path`** or **`--show`** to change the output.

**Example** — pKa predictions on a generated dataset, excluding CYS and TYR, then plotting:

```bash
cd src
python predict.py --pka --atomic --shift --numpy --show-label /path/to/dataset/*.npz | grep -v "CYS" | grep -v "TYR" | python plot_predictions.py
```

To save the figure under a different path: add `-o figure.png` after `plot_predictions.py`. Use `python predict.py -h` and `python plot_predictions.py -h` for all options.

## More info

For more info, or if you have any questions, please email me at **hey@spencerwozniak.com**

## Directory Structure

Here’s a brief overview of the directory structure:

```
ProteinStructureEmbedding/
│
├── pKa-datasets/       # Our datasets for pKa training.
|   |
|   ├── MSU-pKa-train.csv   # Our training dataset.
|   ├── MSU-pKa-val.csv     # Our validation dataset.
|   └── MSU-pKa-test.csv    # Our test dataset.
|
├── src/                # Source code of our application.
|   |
|   ├── dataset.py          # Classes for processing PDBs and generating datasets
|   ├── generate_datasets.py # Script to generate GSnet/aLCnet datasets from residue-level CSVs
|   ├── net.py              # Neural network architectures used in our project
|   ├── predict.py          # Script for making predictions.
|   ├── plot_predictions.py # Script to plot predicted vs observed from predict.py output.
|   ├── embed_GSnet.py      # Script that generates GSnet embeddings.
|   ├── embed_aLCnet.py     # Script that generates aLCnet embeddings.
|   ├── train_GSnet.py      # Script for training GSnet.
|   ├── train_aLCnet.py     # Script for training aLCnet.
|   └── time.sh             # Script for timing the running of a script.
|
├── models/             # State dictionaries containing weights and biases of the models
|   |
|   ├── GSnet_default.pt    # Original pretrained GSnet.
|   ├── GSnet_SASA.pt       # GSnet fine-tuned for SASA predictions.
|   ├── GSnet_pKa.pt        # GSnet fine-tuned for pKa predictions.
|   ├── aLCnet_pKa.pt       # aLCnet trained for pKa predictions.
|   └── normalization.npz   # Normalization parameters.
|
├── sample_data/        # Sample data provided for running certain scripts
|   |
|   ├── time_test/          # Directory containing PDB structures used to test the speed of GSnet
|   ├── GSnet/              # Directory containing sample training and test sets for retraining GSnet.
|   └── aLCnet/             # Directory containing sample training and test sets for retraining aLCnet.
|
├── requirements.txt    # Required Python packages to run our model.
└── README.md           # The file you are currently reading.
```
