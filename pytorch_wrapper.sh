#!/bin/bash

eval "$(/home/$(whoami)/miniconda3/bin/conda shell.bash hook)"

conda activate pytorch

python3 Notebooks/conditional_diffusion_model_unet_pytorch_non_linear.py
