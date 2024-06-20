# FeigLab

import os
import sys
from setuptools import setup, find_packages

name='GNN_testing'

package_dir = {name: name}

setup(
    name=name,
    version='0.1',
    author='Spencer Wozniak',
    author_email='woznia79@msu.edu',
    description=('See github.com/[...]'),
    packages=find_packages(),
    package_dir=package_dir,
    entry_points={
        'console_scripts': [
            'GNNpredict = src.predict:main',
        ],
    },
    install_requires=[
        'numpy',
        'torch',
        'torch_scatter',
        'torch_geometric',
        'mdtraj',
        'scipy',
        'matplotlib',
        'einops',
        'argparse',
        'fair-esm',
        'biopython',
    ],
    python_requires='>=3.7',
)


