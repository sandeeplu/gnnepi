#!/bin/bash

# Activate the Conda environment

source ~/miniconda3/bin/activate my_alignn

##If rerun, delete these dir
rm -rf A* temp/ output.log


##run the command for ALIGNN code

nohup /home/a4724/miniconda3/envs/my_alignn/bin/train_alignn.py --root_dir "/home/a4724/sandeep/alignn_cal/3D/POSCAR_files" --epochs 300 --batch_size 8 --config "/home/a4724/sandeep/alignn_cal/3D/POSCAR_files/config_example.json" --output_dir=temp > output.log 2>&1 &


echo "Script is running in the background. Check 'output.log' for the output."
