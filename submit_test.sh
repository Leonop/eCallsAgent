#!/bin/bash
#SBATCH --job-name=test-bertTopic        # Job name
#SBATCH --output=eCallsAgent/output/log_files/main.out      # Standard output and error log
#SBATCH --error=eCallsAgent/output/log_files/main.err       # Separate file for error logs
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks-per-node=1        # One task per node
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --partition=gpu72          # Use GPU partition
#SBATCH --qos=gpu                  # Required QOS for GPU partitions
#SBATCH --time=72:00:00           # Set time limit to 72 hours
#SBATCH --mail-type=BEGIN,END,FAIL # Notifications for job events
#SBATCH --mail-user=ZXiao@walton.uark.edu

# Load modules
module purge
module load os/el7
module load gcc/11.2.1
module load cuda/11.7
module load python/miniforge-24.3.0

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate bertopic_env

# Set environment variables
export PYTHONPATH="/scrfs/storage/zichengx/home/Research/AIphaBiz:${PYTHONPATH}"
export CUDA_HOME=/usr/local/cuda-11.7
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Change to project directory
cd /scrfs/storage/zichengx/home/Research/AIphaBiz

# Run the main script
python -m eCallsAgent.main