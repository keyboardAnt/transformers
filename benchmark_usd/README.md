To run the benchmark using the default `accelerate` configuration:
```bash
python benchmark.py
```

To enable offloading to CPU when running on a single node with a single GPU:
```bash
source ../.env
accelerate launch --config_file accelerate_config_single_gpu.yaml benchmark.py
```