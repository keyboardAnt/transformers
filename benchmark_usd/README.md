To install the conda environment:
```bash
conda env create -f environment.yml --verbose
```

### Batch mode
On an LSF cluster, you can run the benchmark as follows:
```bash
./submit.sh
```
The `submit.sh` script will submit the benchmark to the LSF queue. To specify the GPU model, edit the `submit.sh` script and add the `-R` option (e.g., `-R "select[gpumodel=='NVIDIAA100_SXM4']"`).

### Interactive mode
Activate the conda environment:
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