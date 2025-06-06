#!/bin/bash
# Set PYTHONPATH to ensure our modules are found
#SBATCH --job-name=main-bertTopic        # Job name
#SBATCH --output=eCallsAgent/output/log_files/main_%j.out      # Standard output and error log
#SBATCH --error=eCallsAgent/output/log_files/main_%j.err       # Separate file for error logs
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks-per-node=1          # One task per node
#SBATCH --cpus-per-task=16          # Number of CPU cores per task
#SBATCH --gres=gpu:1               # Request 1 GPUs
#SBATCH --partition=agpu72          # Use qgpu72 partition (nodes with 4 GPUs)
#SBATCH --qos=gpu                   # Required QOS for GPU partitions
#SBATCH --time=72:00:00             # Set time limit to 72 hours
#SBATCH --mail-type=BEGIN,END,FAIL  # Notifications for job begin, end, and failure
#SBATCH --mail-user=ZXiao@walton.uark.edu  # Your email address

echo "=== Starting job at $(date) ==="

export CFLAGS="-std=c99"
export CXXFLAGS="-std=c++11"
export NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1
export CUDA_HOME=/share/apps/mf/cuda/11.7
export PATH=$CUDA_HOME/bin:$PATH
export PATH=/share/apps/mf/gcc/11.2.1/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/mf/gcc/11.2.1/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/lib64:/usr/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

module purge
module load os/el7
module load gcc/11.2.1
module load cuda/11.7
module load python/miniforge-24.3.0

source /home/zichengx/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda env remove -n bertopic_env -y
conda create -n bertopic_env python=3.9 -y
conda activate bertopic_env

# Verify NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY
if [ "$NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY" != "1" ]; then
    export NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1
fi

# Navigate to project directory
cd /scrfs/storage/zichengx/home/Research/AIphaBiz/poetry-demov2

# Create README if missing
mkdir -p . && touch README.md

# Install poetry and dependencies
pip install poetry==1.8.2
poetry lock --no-update
poetry install --no-root
# Skip PyTorch-related packages in Poetry to avoid authorization issues
pip install uv
# Manually install PyTorch and compatible transformers/huggingface_hub versions
conda install -c rapidsai -c nvidia -c conda-forge \
  cuml=23.08 cudf=23.08 dask-cuda=23.08 \
  cupy cudatoolkit=11.7 -y

pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
pip install huggingface_hub==0.13.4
pip install transformers==4.27.4

# Run the main script with the patched environment
PYTHONPATH=$(pwd):$PYTHONPATH python -m eCallsAgent.main --embedding_model 9

echo "=== Job completed at $(date) ==="