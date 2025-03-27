#!/bin/bash
# Set PYTHONPATH to ensure our modules are found
#SBATCH --job-name=main-bertTopic        # Job name
#SBATCH --output=eCallsAgent/output/log_files/main_%j.out      # Standard output and error log
#SBATCH --error=eCallsAgent/output/log_files/main_%j.err       # Separate file for error logs
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks-per-node=1          # One task per node
#SBATCH --cpus-per-task=16          # Number of CPU cores per task
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

# Set Python path and run main script
export PYTHONPATH="/scrfs/storage/zichengx/home/Research/AIphaBiz:${PYTHONPATH}"
cd /scrfs/storage/zichengx/home/Research/AIphaBiz
python -m eCallsAgent.main

echo "=== Job completed at $(date) ==="