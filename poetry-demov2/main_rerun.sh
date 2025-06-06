#!/bin/bash
# Set PYTHONPATH to ensure our modules are found
#SBATCH --job-name=main-bertTopic        # Job name
#SBATCH --output=eCallsAgent/output/log_files/main_%j.out      # Standard output and error log
#SBATCH --error=eCallsAgent/output/log_files/main_%j.err       # Separate file for error logs
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks-per-node=1          # One task per node
#SBATCH --cpus-per-task=16          # Number of CPU cores per task
#SBATCH --gres=gpu:1               # Request 1 GPUs
#SBATCH --partition=qgpu72          # Use qgpu72 partition (nodes with 4 GPUs)
#SBATCH --qos=gpu                   # Required QOS for GPU partitions
#SBATCH --time=72:00:00             # Set time limit to 72 hours
#SBATCH --mail-type=BEGIN,END,FAIL  # Notifications for job begin, end, and failure
#SBATCH --mail-user=ZXiao@walton.uark.edu  # Your email address


echo "=== Starting job at $(date) ==="

# 1. Set CUDA environment variables first - IMPORTANT: Set these before any Python or CUDA commands
# Export NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY early and explicitly
export NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1
export CUDA_HOME=/share/apps/mf/cuda/11.7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH  # System libraries first
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

module purge
module load os/el7
module load gcc/11.2.1
module load cuda/11.7
module load python/miniforge-24.3.0
source /home/zichengx/miniconda3/etc/profile.d/conda.sh

conda activate bertopic_env

# Force set the compatibility flag again after all installations
export NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1

# 6. Set final library paths & Set environment variables for Python
export PYTHONPATH=$PYTHONPATH:/scrfs/storage/zichengx/home/Research/AIphaBiz/poetry-demov2
export CUDA_MODULE_LOADED=1  # Flag for Python to know CUDA is already loaded

# Verify CUDA
echo "=== Verifying CUDA setup ==="
nvidia-smi || { echo "ERROR: nvidia-smi failed. Exiting."; exit 1; }
which nvidia-smi
nvidia-smi || echo "nvidia-smi failed"
# Verify environment variable is set
echo "NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=$NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY"

echo "=== Checking NVIDIA driver ==="
nvidia-smi || echo "nvidia-smi failed"

echo "=== Checking CUDA runtime ==="
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || echo "Failed to check torch CUDA"

echo "=== Checking RAPIDS components ==="
python -c "import cupy; print(f'CuPy version: {cupy.__version__}')" || echo "Failed to import cupy"
python -c "import cuml; print(f'cuML version: {cuml.__version__}')" || echo "Failed to import cuml"
python -c "import cudf; print(f'cuDF version: {cudf.__version__}')" || echo "Failed to import cudf"
python -c "import dask, dask_cuda; print('Dask and dask_cuda imported successfully')" || echo "Failed to import dask components"
python -c "import nltk, spacy, openai; print('NLP packages imported successfully')" || echo "Failed to import NLP packages"

# Run the main script
echo "=== Running main script ==="
# Set the environment variable in multiple ways to ensure it's recognized
export NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1

# Method 1: Direct environment passing
env NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1 python -m eCallsAgent.main --embedding_model 9

echo "=== Job completed at $(date) ==="