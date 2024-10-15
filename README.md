
# Accurate Predictions of Molecular Properties of Proteins via Graph Neural Networks and Transfer Learning

## Overview

This document offers an overview of the project, installation steps, and instructions to help users effectively implement and utilize our tools and models.

### Graph Neural Network-Based Prediction

This project leverages Graph Neural Networks (GNNs) to accelerate the computational prediction of protein properties. Our model, the `Global Structure Embedding Network (GSnet)`, is adept at predicting a variety of physicochemical properties from three-dimensional protein structures. These properties include:

- Free energy of solvation ($\DeltaG_sol$)

### Transfer Learning for pKa Predictions

A notable feature of `GSnet`, and the related `a-GSnet`, is their ability to deliver rapid and accurate predictions of experimental pKa values, achieved by pretraining on related properties and simulated pKa values, respectively. The application of transfer learning allows these models to utilize previously learned representations, enhancing their predictive accuracy even with limited specific training data. 

### Paper

You can access the pre-print at the following link:

[Enhancing Protein Analysis via Transfer Learning with Graph Neural Networks](https://arxiv.org/)

To cite our pre-print in your work, please use the following citation format:

Wozniak, S., Janson, G., & Feig, M. (2024). Enhancing Protein Analysis via Transfer Learning with Graph Neural Networks. *Preprint available on arXiv*. https://arxiv.org/

## Table of Contents

- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Making Predictions](#making-predictions)
- [Pretrained Models](#pretrained-models)
- [Generating Embeddings](#generating-embeddings)
- [Generating Datasets](#generating-datasets)
- [Implementing our Networks](#implementing-our-networks)
- [Contributing](#contributing)
- [License](#license)
- [To do](#to-do)

## Directory Structure

Here’s a brief overview of the directory structure:

```
hydropro_ml/
│
├── datasets/           # Our datasets.
|   |
|   ├── train.0.csv     # Our training dataset.
|   ├── val.0.csv       # Our validation dataset.
|   └── test.0.csv      # Our test dataset.
|
├── src/                # Source code of our application.
|   |
|   ├── dataset.py      # Classes for processing PDBs and generating datasets
|   ├── net.py          # Neural network architectures used in our project
|   ├── embed.py        # Script that generates embeddings.
|   ├── predict.py      # Script that makes predictions.
|   └── evaluate.py     # Script for evaluating model performance.
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

2. **Install required packages:**
   - Ensure that Python 3.8 or newer is installed on your system.
   - Install the required Python packages using:
     ```bash
     pip install -r requirements.txt
     ```

3. **Set up the environment:**
   - If you use a virtual environment, set it up and activate it before installing the packages:
     ```
     python -m venv venv
     source venv/bin/activate  # On Windows use `venv\Scripts\activate`
     ```

## Making Predictions

### Overview

This section explains how to use our models to make predictions on protein properties such as pKa values and other physicochemical properties from protein structures. The steps below include how to run the script with different options to predict specific properties.

### Steps to Make Predictions

1. **Prepare your environment:**
 Ensure your environment is properly set up as described in the [installation section](#installation), and you have the necessary [pre-trained models](#pretrained-models) downloaded.

2. **Run the prediction script:**
 Use the command below, substituting `<path_to_pdb_file>` with your PDB file's path:
 ```
 python predict.py --option <path_to_pdb_file>
 ```
 Replace `--option` with either `--pka`, `--sasa`, or no option for default predictions.

#### pKa Prediction
- **pKa value prediction:**
```
python predict.py --pka pdb_file.pdb
```
- **Sample Output:**
```
...
7.315245148533568 LYS 4 A /feig/s1/spencer/gnn/cases/groel/1AON_A.pdb
3.9241322437930055 ASP 5 A /feig/s1/spencer/gnn/cases/groel/1AON_A.pdb
8.401511664982062 LYS 7 A /feig/s1/spencer/gnn/cases/groel/1AON_A.pdb
3.903559068776328 ASP 11 A /feig/s1/spencer/gnn/cases/groel/1AON_A.pdb
...
```
- **pKa shift prediction:**
```
python predict.py --pka --shift pdb_file.pdb
```
- **Sample Output:**
```
...
-3.2247548514664315 LYS 4 A /feig/s1/spencer/gnn/cases/groel/1AON_A.pdb
0.024132243793005603 ASP 5 A /feig/s1/spencer/gnn/cases/groel/1AON_A.pdb
-2.1384883350179376 LYS 7 A /feig/s1/spencer/gnn/cases/groel/1AON_A.pdb
0.003559068776328278 ASP 11 A /feig/s1/spencer/gnn/cases/groel/1AON_A.pdb
...
```

#### Default Physicochemical Properties Prediction
- **Command:**
```
python predict.py pdb_file.pdb
```
- **Sample Output:**
```
ΔG [kJ/mol]   RG [Å]        RH [Å]        DT [nm^2/ns]  DR [ns^-1]    V [nm^3]      FILE
-27.580E      12.391E       22.216E       0.982E        0.734E        38.517E       pdb_file.pdb
```

#### Solvent-Accessible Surface Area (SASA) Prediction
- **Command:**
```
python predict.py --sasa pdb_file.pdb
```
- **Sample Output:**
```
ΔG [kJ/mol]   RG [Å]        RH [Å]        DT [nm^2/ns]  DR [ns^-1]    V [nm^3]      SASA [nm^2]   FILE
-27.580E      12.391E       22.216E       0.982E        0.734E        38.517E       57.129E       pdb_file.pdb
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

## Pretrained Models

`...`

## Generating Embeddings

### Overview

This section describes how to generate embeddings for all PDB files within a specified directory. Embeddings are crucial for downstream prediction tasks and are saved for subsequent use.

### Steps to Generate Embeddings

1. **Prepare your environment:**
 Ensure you have followed the installation instructions to set up your environment correctly. This includes having the necessary Python packages installed and the environment activated.

2. **Obtain the model:**
 Currently, you will need to manually download our pre-trained model `pka_from_sasa_res.pt` from the provided link (link to be added). Place this model in an appropriate directory accessible to your script.

3. **Run the script:**
 Navigate to the `src/` directory in your terminal. Use the following command to generate embeddings:
 ```
 python embed.py <path_to_PDB_files> <output_path_for_embeddings>
 ```
 Replace `<path_to_PDB_files>` with the directory containing your PDB files and `<output_path_for_embeddings>` with the directory where you want to save the embeddings.

### Example Command

```
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
- The embedding process is sensitive to the structure and quality of the input PDB files. Ensure that the files are correctly formatted and contain all necessary information.

This method of generating embeddings is integral to leveraging the predictive power of our GNN models for analyzing protein structures.

## Generating Datasets

`...`

## Implementing our Networks

`...`

## Contributing

Contributions to this project are welcome! Please refer to our contributing guidelines for more information on how to submit pull requests, report issues, and participate in code reviews.

## License

This project is not yet licensed - see the [LICENSE](LICENSE) file for details.

## To do

- Add pretrained models (will be large files though, may need to upload elsewhere)
- Polish up and add training scripts
- Anything else??
