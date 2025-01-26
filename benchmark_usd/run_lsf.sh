#!/bin/bash
#BSUB -q long-gpu
#BSUB -M 64GB
#BSUB -gpu "num=1:mode=exclusive_process:gmem=80000"
#BSUB -oo /home/projects/dharel/nadavt/repos/transformers/benchmark_usd/lsf_logs/$(date +\%Y\%m\%d_\%H\%M\%S)_jobid_%J_benchmark_out.log
#BSUB -eo /home/projects/dharel/nadavt/repos/transformers/benchmark_usd/lsf_logs/$(date +\%Y\%m\%d_\%H\%M\%S)_jobid_%J_benchmark_err.log

module load miniconda/24.11_environmentally
conda activate bench-env
python /home/projects/dharel/nadavt/repos/transformers/benchmark_usd/benchmark.py
