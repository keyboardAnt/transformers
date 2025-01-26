To install the conda environment:
```bash
conda env create -f environment.yml --verbose
```
And activate the environment:
```bash
conda activate bench-env
```

To run the benchmark using the default `accelerate` configuration:
```bash
python benchmark.py
```

<!-- To enable offloading to CPU when running on a single node with a single GPU:
```bash
source ../.env
accelerate launch --config_file accelerate_config_single_gpu.yaml benchmark.py
``` -->

On an LSF cluster, you can run the benchmark manually as follows. Activate the conda environment and call the benchmark script.
```bash
timestamp=$(date +\%Y\%m\%d_\%H\%M\%S) && bsub -q "long-gpu short-gpu risk-gpu" \
     -M 64GB \
     -gpu "num=1:mode=exclusive_process:gmem=80000" \
     -oo /home/projects/dharel/nadavt/repos/transformers/benchmark_usd/lsf_logs/${timestamp}_jobid_%J_benchmark_out.log \
     -eo /home/projects/dharel/nadavt/repos/transformers/benchmark_usd/lsf_logs/${timestamp}_jobid_%J_benchmark_err.log \
     "module load miniconda/24.11_environmentally && conda activate bench-env && python /home/projects/dharel/nadavt/repos/transformers/benchmark_usd/benchmark.py"
```
To specify the GPU model, add the `-R` option. For example:
```bash
-R "select[gpumodel=='NVIDIAA100_SXM4']"
```