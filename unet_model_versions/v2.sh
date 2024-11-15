#!/bin/bash

eval "$(/home/$(whoami)/miniconda3/bin/conda shell.bash hook)"

conda activate pytorch

python3 unet_model_versions/v2.py
