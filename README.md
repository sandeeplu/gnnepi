**Graph Neural Networks for Electrophilicity Index Prediction**

This repository contains the code associated with the manuscript. In this manuscript, we compare 3D molecular geometry models (SchNet,ALIGNN, GemNet), which account for the full atomic structure, with connectivitybased
2D models (Attentive FP, GCN, GraphSAGE, GIN, GINE, GATv2) that consider only molecular topology. These GNNS models are trained for Electrophilicity Index Prediction on the QM9 dataset.

The data for training the models are taken from QM9:
https://quantum-machine.org/datasets/

The libraries required for running the code are :
Data analysis (Pandas, Matplotlib/Seaborn, RDKit)
Generate graphs from SMILES (Pytorch Geometric)
Graph Analysis (Networkx)
Train Test Split (Pytorch)
Defining Functions for Training and Evaluation
Model Training and Evaluation (Pytorch Geometric)
Hyperparameter Tuning (Optuna)
