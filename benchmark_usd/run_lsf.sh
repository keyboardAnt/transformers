#!/bin/bash
#BSUB -q long-gpu
#BSUB -gpu "num=1:gmem=40000"

module load miniconda/24.11_environmentally
conda activate benchmark-usd-env
python benchmark.py