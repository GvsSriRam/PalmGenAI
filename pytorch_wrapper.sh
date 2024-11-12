#!/bin/bash

eval "$(/home/$(whoami)/miniconda3/bin/conda shell.bash hook)"

conda activate pytorch

# conda install --yes --file requirements.txt
pip install -r requirements.txt

python3 Notebooks/conditional_diffusion_model_unet_pytorch.py
