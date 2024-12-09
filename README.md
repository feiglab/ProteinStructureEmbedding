# Accurate Predictions of Molecular Properties of Proteins via Graph Neural Networks and Transfer Learning

## Overview

### Graph Neural Network-Based Prediction

This project leverages **Graph Neural Networks (GNNs)** to accelerate the computational prediction of protein properties. Our model, the `Global Structure Embedding Network (GSnet)`, is adept at predicting a variety of physicochemical properties from three-dimensional protein structures. A notable feature of `GSnet`, and the related `aLCnet`, is their ability to deliver rapid and accurate predictions of experimental pKa values, achieved by pretraining on related properties and simulated pKa values, respectively. The application of **transfer learning** allows these models to utilize previously learned representations, enhancing their predictive accuracy even with limited specific training data. 

Properties that these models can predict include:

- Free energy of solvation ($\Delta G_{sol}$)
- Hydrodynamic radius ($R_h$)
- Translational diffusion coefficient ($D_t$)
- Rotational diffusion coefficient ($D_r$)
- Molecular volume ($V$)
- Radius of gyration ($R_g$)
- Solvent accessible surface area ($SASA$)
- $pK_a$ values

### Paper

You can access the pre-print at the following link:

[Accurate Predictions of Molecular Properties of Proteins via Graph Neural Networks and Transfer Learning](https://arxiv.org/)

To cite our pre-print in your work, use:

Wozniak, S., Janson, G., & Feig, M. (2024). Accurate Predictions of Molecular Properties of Proteins via Graph Neural Networks and Transfer Learning. *Preprint available on arXiv*. https://arxiv.org/

## Table of Contents

- [Installation](#installation)
- [Making Predictions](#making-predictions)
- [Pretrained Models](#pretrained-models)
- [Generating Embeddings](#generating-embeddings)
- [Implementing our Networks](#implementing-our-networks)
- [Generating Datasets](#generating-datasets)
- [Directory Structure](#directory-structure)



## Installation

Before you can run the models and use the codebase, you need to set up your environment:

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

This section explains how to use our models to make predictions on protein properties such as pKa values and other physicochemical properties from protein structures. The steps below include how to run the script with different options to predict specific properties.

### Steps to Make Predictions

1. **Prepare your environment:**
 Ensure your environment is properly set up as described in the [installation section](#installation), and you have the necessary [pre-trained models](#pretrained-models) downloaded.

2. **Run the prediction script:**

#### Default Physicochemical Properties Prediction
- **Command:**
```bash
python predict.py /path/to/pdb_file.pdb
```
- **Sample Output:**
```
ΔG [kJ/mol]   RG [Å]        RH [Å]        DT [nm^2/ns]  DR [ns^-1]    V [nm^3]      FILE
-2.888315E+04 2.9318836E+01 3.4585515E+01 6.2096068E-02 3.7419807E-03 9.2364510E+01 /path/to/pdb_file.pdb
```

#### Solvent-Accessible Surface Area (SASA) Prediction
- **Command:**
```bash
python predict.py --sasa /path/to/pdb_file.pdb
```
- **Sample Output:**
```
ΔG [kJ/mol]   RG [Å]        RH [Å]        DT [nm^2/ns]  DR [ns^-1]    V [nm^3]      SASA [nm^2]   FILE
-2.888315E+04 2.9318836E+01 3.4585515E+01 6.2096068E-02 3.7419807E-03 9.2364510E+01 2.3868515E+02 /path/to/pdb_file.pdb
```

#### pKa Prediction
**pKa value prediction:**
- **Command:**
```bash
python predict.py --pka /path/to/pdb_file.pdb
```
Note that `GSnet` is the default option. pKa predictions are faster and more accurate with `aLCnet`. To use `aLCnet`, you must use the option `--atomic`.
```bash
python predict.py --pka --atomic /path/to/pdb_file.pdb
```
- **Sample Output:**
```
7.315245148533568 LYS 4 A /path/to/pdb_file.pdb
3.9241322437930055 ASP 5 A /path/to/pdb_file.pdb
8.401511664982062 LYS 7 A /path/to/pdb_file.pdb
3.903559068776328 ASP 11 A /path/to/pdb_file.pdb
...
```
**pKa shift prediction**
- **Command:**
```bash
python predict.py --pka --shift /path/to/pdb_file.pdb
```
or
```bash
python predict.py --pka --shift --atomic /path/to/pdb_file.pdb
```
- **Sample Output:**
```
-3.2247548514664315 LYS 4 A /path/to/pdb_file.pdb
0.024132243793005603 ASP 5 A /path/to/pdb_file.pdb
-2.1384883350179376 LYS 7 A /path/to/pdb_file.pdb
0.003559068776328278 ASP 11 A /path/to/pdb_file.pdb
...
```

### Notes

- The script is capable of handling multiple input files and can process them in batches if specified.
- For detailed error messages and troubleshooting, the script outputs logs that can be checked in case of failures.
- Remember to use the cleaning options if your PDB files might contain non-standard residues or formats.

For a full description of the utility of the `predict.py` script, you can run `python predict.py -h` or `python predict.py --help`:

```
usage: predict.py [-h] [--clean] [--pka] [--atomic] [--sasa] [--shift] [--chain chain] [--combine-chains] [--keep] [--cpu] [--gpu] [--numpy] [--time]
                  [--skip-bad-files]
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

2. **Run the script:**
 Navigate to the `src/` directory in your terminal. Use the following command to generate embeddings:
 ```bash
 python embed.py --protein/--residue <path_to_PDB_files> <output_path_for_embeddings>
 ```
 Replace `<path_to_PDB_files>` with the directory containing your PDB files and `<output_path_for_embeddings>` with the directory where you want to save the embeddings.

- Use the `--protein` option to generate GSnet embeddings optimized for **whole protein predictions** (trained on 6 physicochemical properties).
- Use the `--residue` option to generate GSnet embeddings optimized for **residue-specific predictions** (fine-tuned on rSASA and pKa).

### Notes

- The script utilizes multiprocessing to expedite the embedding process. Ensure your system has adequate resources to handle multiple processes simultaneously.
- In theory, either embedding method (`--protein` or `--residue`) may be useful in either context. It could be worthwhile to try both embeddings for the same task to determine which is more useful.

## More info

For more info, please email me at **spencerwozniak1@gmail.com**

## Directory Structure

Here’s a brief overview of the directory structure:

```
hydropro_ml/
│
├── datasets/           # Our datasets.
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
|   └── train_aLCnet.py     # Script for training aLCnet.
|
├── models/             # State dictionaries containing weights and biases of the models
|   |
|   ├── GSnet_default.pt    # Original pretrained GSnet.
|   ├── GSnet_SASA.pt       # GSnet fine-tuned for SASA predictions.
|   ├── GSnet_pKa.pt        # GSnet fine-tuned for pKa predictions.
|   ├── aLCnet_pka.pt       # aLCnet trained for pKa predictions.
|   └── normalization.npz   # Normalization parameters.
|
├── requirements.txt    # Required Python packages to run our model.
├── setup.py            # Script for installing our model as a package.
└── README.md           # The file you are currently reading.
```
