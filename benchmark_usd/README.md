To run the benchmark using the default `accelerate` configuration:
```bash
python benchmark.py
```

<!-- To enable offloading to CPU when running on a single node with a single GPU:
```bash
source ../.env
accelerate launch --config_file accelerate_config_single_gpu.yaml benchmark.py
``` -->

On an LSF cluster, you can run the benchmark manually as follows:
1. Activate the conda environment:
```bash
conda activate benchmark-usd-env
```
2. Run the benchmark:
```bash
bsub -q "long-gpu short-gpu risk-gpu" \
     -gpu "num=1:gmem=80000" \
     -R "select[gpumodel=='NVIDIAA100_SXM4']" \
     -oo benchmark_output_%J.log \
     -eo benchmark_error_%J.log \
     "source /home/projects/dharel/nadavt/env/bin/activate && python /home/projects/dharel/nadavt/repos/transformers/benchmark_usd/benchmark.py"
```
