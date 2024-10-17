# Accurate Predictions of Molecular Properties of Proteins via Graph Neural Networks and Transfer Learning

**GSnet**
![GSNet architecture](https://github.com/user-attachments/assets/2e49b0b6-a74b-4481-9a8d-129edeb0e57a)

**a-GSnet**
![a-GSnet architecture](https://github.com/user-attachments/assets/d74130a7-120a-4542-8cfd-19041936cdf8)

## Overview

This document offers an overview of the project, installation steps, and instructions to help users effectively implement and utilize our tools and models.

### Graph Neural Network-Based Prediction

This project leverages **Graph Neural Networks (GNNs)** to accelerate the computational prediction of protein properties. Our model, the `Global Structure Embedding Network (GSnet)`, is adept at predicting a variety of physicochemical properties from three-dimensional protein structures. A notable feature of `GSnet`, and the related `a-GSnet`, is their ability to deliver rapid and accurate predictions of experimental pKa values, achieved by pretraining on related properties and simulated pKa values, respectively. The application of **transfer learning** allows these models to utilize previously learned representations, enhancing their predictive accuracy even with limited specific training data. 

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

- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Making Predictions](#making-predictions)
- [Pretrained Models](#pretrained-models)
- [Generating Embeddings](#generating-embeddings)
- [Generating Datasets](#generating-datasets)
- [Implementing our Networks](#implementing-our-networks)
- [To do](#to-do)

## Directory Structure

Here’s a brief overview of the directory structure:

```
hydropro_ml/
│
├── datasets/           # Our datasets.
|   |
|   ├── train.0.csv         # Our training dataset.
|   ├── val.0.csv           # Our validation dataset.
|   └── test.0.csv          # Our test dataset.
|
├── src/                # Source code of our application.
|   |
|   ├── dataset.py          # Classes for processing PDBs and generating datasets
|   ├── net.py              # Neural network architectures used in our project
|   ├── embed.py            # Script that generates embeddings.
|   └── predict.py          # Script for making predictions.
|
├── models/             # State dictionaries containing weights and biases of the models
|   |
|   ├── GSnet_default.pt    # Original pretrained GSnet.
|   ├── GSnet_SASA.pt       # GSnet fine-tuned for SASA predictions.
|   ├── GSnet_pKa.pt        # GSnet fine-tuned for pKa predictions.
|   ├── aGSnet_pka.pt       # aGSnet trained for pKa predictions.
|   └── normalization.npz   # Normalization parameters.
|
├── requirements.txt    # Required Python packages to run our model.
├── setup.py            # Script for installing our model as a package.
└── README.md           # The file you are currently reading.
```

## Installation

Before you can run the models and use the codebase, you need to set up your environment:

1. **Clone the repository:**

```bash
git clone https://github.com/your-repository/hydropro_ml.git
cd hydropro_ml
```

2. **Set up the environment:**
   - If you use a virtual environment, set it up and activate it before installing the packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
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
 Use the command below, substituting `<path_to_pdb_file>` with your PDB file's path:
 ```bash
 python predict.py --option <path_to_pdb_file>
 ```
 Replace `--option` with either `--pka`, `--sasa`, or no option for default predictions.
 
 

#### pKa Prediction
- **pKa value prediction:**
```bash
python predict.py --pka /path/to/pdb_file.pdb
```
Note that `GSnet` is the default option. pKa predictions are more accurate with `a-GSnet`. To use `a-GSnet`, you must use the option `--atomic`.
```bash
python predict.py --pka --atomic /path/to/pdb_file.pdb
```
- **Sample Output:**
```
...
7.315245148533568 LYS 4 A /path/to/pdb_file.pdb
3.9241322437930055 ASP 5 A /path/to/pdb_file.pdb
8.401511664982062 LYS 7 A /path/to/pdb_file.pdb
3.903559068776328 ASP 11 A /path/to/pdb_file.pdb
...
```
- **pKa shift prediction:**
```bash
python predict.py --pka --shift /path/to/pdb_file.pdb
```
or
```bash
python predict.py --pka --shift --atomic /path/to/pdb_file.pdb
```
- **Sample Output:**
```
...
-3.2247548514664315 LYS 4 A /path/to/pdb_file.pdb
0.024132243793005603 ASP 5 A /path/to/pdb_file.pdb
-2.1384883350179376 LYS 7 A /path/to/pdb_file.pdb
0.003559068776328278 ASP 11 A /path/to/pdb_file.pdb
...
```

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

### Troubleshooting

- Ensure all paths are correct and the necessary files are accessible.
- Verify that the pre-trained model files are in the correct directory and are properly named.
- Check your Python environment if you encounter dependency errors.
- If the script fails with a CUDA error and you are running on a GPU, make sure your device has sufficient memory and is compatible.

### Notes

- The script is capable of handling multiple input files and can process them in batches if specified.
- For detailed error messages and troubleshooting, the script outputs logs that can be checked in case of failures.
- Remember to use the cleaning options if your PDB files might contain non-standard residues or formats.

These methods allow for the flexible application of our models to a variety of prediction tasks in protein analysis.

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
  --atomic          Use a-GSnet for pKa predictions
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
| `aGSnet_pKa.pt`       | 4,784,324       | aGSnet trained from scratch on PHMD549 data and fine-tuned for pKa prediction on experimental data. |


## Generating Embeddings

### Overview

This section describes how to generate embeddings for all PDB files within a specified directory. Embeddings are crucial for downstream prediction tasks and are saved for subsequent use.

### Steps to Generate Embeddings

1. **Prepare your environment:**
 Ensure you have followed the installation instructions to set up your environment correctly. This includes having the necessary Python packages installed and the environment activated.

2. **Run the script:**
 Navigate to the `src/` directory in your terminal. Use the following command to generate embeddings:
 ```bash
 python embed.py <path_to_PDB_files> <output_path_for_embeddings>
 ```
 Replace `<path_to_PDB_files>` with the directory containing your PDB files and `<output_path_for_embeddings>` with the directory where you want to save the embeddings.

### Example Command

```bash
python embed.py /path/to/pdb /path/to/output
```

This command processes all PDB files in `/path/to/pdb` and saves the resulting embeddings in `/path/to/output`.

### Troubleshooting

- Ensure that the paths provided are correct and accessible.
- Verify that the pre-trained model file is in the correct directory and properly named.
- Check your Python environment if you encounter dependencies errors.
- If the script fails with a CUDA error and you are running on a GPU, ensure that your device has sufficient memory and is compatible.

### Notes

- The script utilizes multiprocessing to expedite the embedding process. Ensure your system has adequate resources to handle multiple processes simultaneously.
- The embedding process can be sensitive to the structure and quality of the input PDB files. The scripts will automatically attempt to clean and process broken PDB files, but it may not always work. Please ensure that the files you use are correctly formatted and contain all necessary information.

This method of generating embeddings is integral to leveraging the predictive power of our GNN models for analyzing protein structures.

## Generating Datasets

`...`

## Implementing our Networks

`...`

## To do

- Polish up and add training scripts
- Anything else??
