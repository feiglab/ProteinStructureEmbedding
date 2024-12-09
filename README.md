# Accurate Predictions of Molecular Properties of Proteins via Graph Neural Networks and Transfer Learning

## Overview

### Graph Neural Network-Based Prediction

This repository contains code and pre-trained weights for **Graph Transformer Networks** that are trained for the computational prediction of protein properties. [GSnet](#gsnet) is adept at predicting a variety of physicochemical properties from three-dimensional protein structures, while [aLCnet](#alcnet) was specifically trained for residue-specific pKa prediction. Moreover, the application of **transfer learning** allows these models to utilize previously learned representations (_i.e._, embeddings) in new prediction tasks, even with limited specific training data.

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

![aLCnet architecture dark](https://github.com/user-attachments/assets/727838b7-8e88-43cb-92e9-b8588a069634)
</details>

<details><summary><b>Paper</b></summary>

You can access the pre-print at the following link:

[Accurate Predictions of Molecular Properties of Proteins via Graph Neural Networks and Transfer Learning](https://arxiv.org/)

To cite our pre-print in your work, use:

Wozniak, S., Janson, G., & Feig, M. (2024). Accurate Predictions of Molecular Properties of Proteins via Graph Neural Networks and Transfer Learning. *Preprint available on arXiv*. https://arxiv.org/
</details>


<details open><summary><b>Table of Contents</b></summary>

- [Installation](#installation)
- [Making Predictions](#making-predictions)
- [Pretrained Models](#pretrained-models)
- [Generating Embeddings](#generating-embeddings)
- [Generating Datasets](#generating-datasets)
- [Training a Model](#training-a-model)
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
conda create -n gsnet
conda activate gsnet
```

3. **Install required packages:**
   - Ensure that Python 3.8 or newer is installed on your system.
   - Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Making Predictions

### Overview

This section explains how to use our models to make predictions of protein properties such as pKa values and other physicochemical properties from protein structures. The steps below include how to run the script with different options to predict specific properties.

### Steps to Make Predictions

1. **Prepare your environment:** Ensure your environment is properly set up as described in the [installation section](#installation), and you have the necessary [pre-trained models](#pretrained-models) downloaded.

2. **Run the prediction script:**

<details><summary><b>Default Physicochemical Properties Prediction</b></summary>

**Command:**

```bash
python predict.py /path/to/pdb_file.pdb
```
    
**Sample Output:**

```
ΔG [kJ/mol]   RG [Å]        RH [Å]        DT [nm^2/ns]  DR [ns^-1]    V [nm^3]      FILE
-2.888315E+04 2.9318836E+01 3.4585515E+01 6.2096068E-02 3.7419807E-03 9.2364510E+01 /path/to/pdb_file.pdb
```

</details>

<details><summary><b>Solvent-Accessible Surface Area (SASA) Prediction</b></summary>

**Command:**

```bash
python predict.py --sasa /path/to/pdb_file.pdb
```

**Sample Output:**

```
ΔG [kJ/mol]   RG [Å]        RH [Å]        DT [nm^2/ns]  DR [ns^-1]    V [nm^3]      SASA [nm^2]   FILE
-2.888315E+04 2.9318836E+01 3.4585515E+01 6.2096068E-02 3.7419807E-03 9.2364510E+01 2.3868515E+02 /path/to/pdb_file.pdb
```

</details>

<details><summary><b>pKa value prediction</b></summary>

**Command:**

Note that  pKa predictions are faster and more accurate with `aLCnet`. To use `aLCnet`, you must use the option `--atomic`.

```bash
python predict.py --pka --atomic /path/to/pdb_file.pdb
```

**Sample Output:**
```
7.315245148533568 LYS 4 A /path/to/pdb_file.pdb
3.9241322437930055 ASP 5 A /path/to/pdb_file.pdb
8.401511664982062 LYS 7 A /path/to/pdb_file.pdb
3.903559068776328 ASP 11 A /path/to/pdb_file.pdb
...
```
</details>

<details><summary><b>pKa shift prediction</b></summary>

**Command:**

```bash
python predict.py --pka --shift --atomic /path/to/pdb_file.pdb
```

**Sample Output:**

```
-3.2247548514664315 LYS 4 A /path/to/pdb_file.pdb
0.024132243793005603 ASP 5 A /path/to/pdb_file.pdb
-2.1384883350179376 LYS 7 A /path/to/pdb_file.pdb
0.003559068776328278 ASP 11 A /path/to/pdb_file.pdb
...
```

</details>


### Notes

- The script is capable of handling multiple input files if you provide them as arguments.
- Remember to use the cleaning options if your PDB files might contain non-standard residues or formats.

For a full description of the utility of the `predict.py` script, you can run `python predict.py -h` or `python predict.py --help`:

```
usage: predict.py [-h] [--clean] [--pka] [--atomic] [--sasa] [--shift]
                  [--chain chain] [--combine-chains] [--keep]
                  [--cpu] [--gpu] [--numpy] [--time] [--skip-bad-files]
                  pdbs [pdbs ...]

ML prediction on PDB files

positional arguments:
  pdbs              List of PDB files.

optional arguments:
  -h, --help        show this help message and exit
  --clean           Clean PDB files before making predictions.
  --pka             Predict pKa.
  --atomic          Use aLCnet for pKa predictions
  --sasa            Predict SASA.
  --shift           Calculate pKa shift (relative to standard value).
  --chain chain     Specify chain.
  --combine-chains  Make calculation for structure of all chains in a PDB file.
  --keep            Keep cleaned PDB files.
  --cpu             Run on CPU.
  --gpu             Run on GPU.
  --numpy           Use .npz file as input.
  --time            Time different aspects of the model.
  --skip-bad-files  Skip bad PDB files.
```

## Pretrained Models

| Model Name           | Number of Parameters | Description                                                                                     |
|----------------------|----------------------|-------------------------------------------------------------------------------------------------|
| `GSnet_default.pt`    | 5,971,748       | The original GSnet model trained on the 6 physicochemical properties. |
| `GSnet_SASA.pt`       | 5,971,748       | GSnet fine-tuned for molecular SASA predictions.                  |
| `GSnet_pKa.pt`        | 11,210,392       | GSnet fine-tuned for residue-level SASA, then further trained to predict pKa values.              |
| `aLCnet_pKa.pt`       | 4,784,324       | aLCnet trained from scratch on PHMD549 data and fine-tuned for pKa prediction on experimental data. |


## Generating Embeddings

### Overview

This section describes how to generate embeddings for all PDB files within a specified directory. Embeddings are crucial for downstream prediction tasks and can be saved for subsequent use.

- Generated embeddings (via either method) will be saved as tensors of shape `[N,d]` where `N` is the number of residues in the protein and `d` is the embedding dimension.

### Steps to Generate Embeddings

1. **Prepare your environment:**
 Ensure you have followed the installation instructions to set up your environment correctly. This includes having the necessary Python packages installed and the environment activated.

2. **Gather PDB files**
 Put PDB files containing only 1 chain that you would like embeddings for into a directory. Make sure the file extension for the files is `.pdb`.
 
3. **Run the script:**
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

The `dataset.py` script allows the creation of datasets for both GSnet and aLCnet via `NumpyRep` and `NumpyRep_atomic` classes, respectively, then via `ProteinDataset` and `AtomicDataset` classes, respectively.

To create a dataset:

#### For GSnet:

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

#### For aLCnet:

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
2. See the `train_GSnet.py` and `train_aLCnet.py` scripts for examples on how to train our models. Sample data for training both GSnet and aLCnet was provided for selected IDP structures.

## More info

For more info, or if you have any questions, please email me at **spencerwozniak1@gmail.com**

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
|   ├── net.py              # Neural network architectures used in our project
|   ├── predict.py          # Script for making predictions.
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
|   ├── aLCnet_pka.pt       # aLCnet trained for pKa predictions.
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
