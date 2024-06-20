
# Enhancing Protein Analysis via Transfer Learning with Graph Neural Networks

## Overview

This project utilizes Graph Neural Networks (GNNs) for enhancing the analysis of protein structures by employing a method called transfer learning. Specifically, our Global Structure Embedding Network (GSnet) is designed to predict several molecular properties of proteins from their three-dimensional structures, including solvation energies and hydrodynamic radii. A notable feature of GSnet is its capability to make rapid and accurate pKa predictions, which are crucial for understanding protein behavior in different environments. This README outlines the project structure, installation steps, and usage instructions to help you get started with our tools and models.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)
- [To do](#to-do)

## Installation

Before you can run the models and use the codebase, you need to set up your environment:

1. **Clone the repository:**
   ```
   git clone https://github.com/your-repository/hydropro_ml.git
   cd hydropro_ml
   ```

2. **Install required packages:**
   - Ensure that Python 3.8 or newer is installed on your system.
   - Install the required Python packages using:
     ```
     pip install -r requirements.txt
     ```

3. **Set up the environment:**
   - If you use a virtual environment, set it up and activate it before installing the packages:
     ```
     python -m venv venv
     source venv/bin/activate  # On Windows use `venv\Scripts\activate`
     ```

## Usage

To utilize our project, follow these steps:

1. **Prepare your data:**
   - Ensure your data is in the correct format as described in the `datasets/` directory.

2. **Training the model:**
   - To train the model with the provided datasets, run:
     ```
     python src/train.py --dataset datasets/train.0.csv
     ```

3. **Generating embeddings:**
   - Generate embeddings for new protein structures:
     ```
     python src/embed.py --input your_protein_data.pdb --output embeddings.csv
     ```

4. **Making predictions:**
   - Use the trained model to predict properties:
     ```
     python src/predict.py --model path_to_model --input embeddings.csv --output predictions.csv
     ```

5. **Evaluating the model:**
   - Evaluate the model's performance with the validation set:
     ```
     python src/evaluate.py --model path_to_model --dataset datasets/val.0.csv
     ```

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

## Contributing

Contributions to this project are welcome! Please refer to our contributing guidelines for more information on how to submit pull requests, report issues, and participate in code reviews.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## To do

- Add pretrained models (will be large files though, may need to upload elsewhere)
- Polish up and add training scripts
- Anything else??
