import subprocess
import os
import yaml
import tempfile
from pathlib import Path
import torch


available_gpus = list(range(torch.cuda.device_count()))

base_config_path = "dinov2/configs/train/mammo-config.yaml"
base_output_root = "outputs"
learning_rates = [2e-4, 2e-5, 2e-6]

if len(available_gpus) != len(learning_rates):
    raise ValueError('Grid search / GPU mismatch!')

processes = []

with open(base_config_path, "r") as f:
    base_config = yaml.safe_load(f)

for i, (lr, gpu_id) in enumerate(zip(learning_rates, available_gpus)):
    cfg = base_config.copy()
    cfg["optim"]["lr"] = lr

    run_name = f"lr-{lr:.0e}".replace("-", "")
    output_dir = Path(base_output_root) / f"{run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write temporary config to output dir
    temp_config_path = output_dir / "temp-config.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(cfg, f)

    port = 29500 + i
    
    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint=localhost:{port}",
        "dinov2/train/train.py",
        "--config-file", str(temp_config_path),
        "--output-dir", str(output_dir)
    ]
    
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
    }

    print(f"Launching training with lr={lr} on GPU {gpu_id}")
    proc = subprocess.Popen(cmd, env=env)
    processes.append((proc, temp_config_path))

# Optional: wait for all to finish and remove temp configs
for proc, temp_config_path in processes:
    proc.wait()
    temp_config_path.unlink()  # delete the temp config file
