\##GCAT-BCE
Graph attention convolutional network for conformational B cell epitope prediction



\##Project Overview
This repository contains complete source code, multiple FASTA datasets, and a pre trained best model for conformational B cell epitope prediction called 'best\_madel\_GCAT-BCE. pt'.



\##File directory
train data:
BP3\_582(pdb).fasta

test data:
BP3\_15(pdb).fasta
PDB2526 28.fasta
RoBep\_187.fasta

best model:
best\_model\_GCAT-BCE.pt

feature generation file:
feature\_dssp.py
feature.py

model file:
GCAT\_BCE.py

training and testing, and individual testing files:
train\_val\_test.py
test.py

Project introduction:

README.md



\## Environment Requirements

python >= 3.8

pytorch >= 1.10

Biopython

numpy

pandas

scikit-learn

DSSP



\## Usage Guide

1\. Prepare protein PDB and FASTA sequence data

2\. Run 'feature\_dssp.py' and 'feature.py' to extract structural features

3\. Train model via 'train\_val\_test.py'

4\. Load 'best\_model\_GCAT-BCE.pt' and run prediction with 'test.py'



\## Datasets

1\. BP3 dataset

2\. RoBep187 dataset

3\. Self-built PDB2526 dataset



\## Citation

If you use this code and dataset in your research, please cite our paper.





