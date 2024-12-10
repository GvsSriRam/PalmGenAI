#!/bin/bash

eval "$(/home/$(whoami)/miniconda3/bin/conda shell.bash hook)"

conda activate pytorch

python3 Notebooks/basic_diffusion_model_128.py
