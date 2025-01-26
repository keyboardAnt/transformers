#! /bin/bash

timestamp=$(date +\%Y\%m\%d_\%H\%M\%S) && bsub -q "long-gpu short-gpu risk-gpu" \
     -M 200GB \
     -gpu "num=1:mode=exclusive_process:gmem=80000" \
     -oo /home/projects/dharel/nadavt/repos/transformers/benchmark_usd/lsf_logs/${timestamp}_jobid_%J_benchmark_out.log \
     -eo /home/projects/dharel/nadavt/repos/transformers/benchmark_usd/lsf_logs/${timestamp}_jobid_%J_benchmark_err.log \
     "module load miniconda/24.11_environmentally && module load CUDA/12.4.0 && conda activate bench-env && python /home/projects/dharel/nadavt/repos/transformers/benchmark_usd/benchmark.py"