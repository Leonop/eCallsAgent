#!/bin/bash
# Set PYTHONPATH to ensure our modules are found
#SBATCH --job-name=main-bertTopic        # Job name
#SBATCH --output=eCallsAgent/output/log_files/main_%j.out      # Standard output and error log
#SBATCH --error=eCallsAgent/output/log_files/main_%j.err       # Separate file for error logs
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks-per-node=1          # One task per node
#SBATCH --cpus-per-task=32          # Number of CPU cores per task
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --partition=agpu72       # Use GPU partition
#SBATCH --qos=gpu               # Required QOS for GPU partitions
#SBATCH --time=72:00:00              # Set time limit to 24 hours
#SBATCH --mail-type=BEGIN,END,FAIL   # Notifications for job begin, end, and failure
#SBATCH --mail-user=ZXiao@walton.uark.edu  # Your email address

echo "=== Starting job at $(date) ==="

# Source conda
source /home/zichengx/miniconda3/etc/profile.d/conda.sh || {
    echo "ERROR: Could not find conda. Exiting."
    exit 1
}

# Activate or create environment
conda activate bertopic_env || {
    echo "Creating new bertopic_env environment..."
    conda create -n bertopic_env python=3.8 -y
    conda activate bertopic_env
}
# pip install -U kaleido
# pip freeze > eCallsAgent/requirements.txt
# pip install poetry
# poetry install
# Verify CUDA
echo "=== Verifying CUDA setup ==="
nvidia-smi || { echo "ERROR: nvidia-smi failed. Exiting."; exit 1; }
python -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: CUDA not available')
    exit(1)
print('CUDA available:', torch.cuda.get_device_name(0))
"

# Before installing cupy-cuda11x, uninstall any existing CuPy installations
echo "Uninstalling any existing CuPy packages..."
$PIP_CMD uninstall -y cupy cupy-cuda11x || echo "No CuPy packages to uninstall"
conda uninstall -y cupy || echo "No CuPy conda package to uninstall"

# Then install only cupy-cuda11x
echo "Installing cupy-cuda11x..."
$PIP_CMD install --no-cache-dir cupy-cuda11x==13.4.1

# 11. Install RAPIDS components using conda rather than pip
echo "Installing RAPIDS components using conda..."
# Make sure this conda install comes BEFORE any pip installs that might pull in cupy
conda install -c rapidsai -c conda-forge -c nvidia \
    cudf=23.8 \
    cuml=23.8 \
    dask-cuda=23.8 \
    dask-cudf=23.8 \
    cudatoolkit=11.7 \
    cupy

# Set Python path and run main script
export PYTHONPATH="/scrfs/storage/zichengx/home/Research/AIphaBiz:${PYTHONPATH}"
cd /scrfs/storage/zichengx/home/Research/AIphaBiz/poetry-demo
poetry run python -m eCallsAgent.main

echo "=== Job completed at $(date) ==="