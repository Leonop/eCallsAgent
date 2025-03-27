#!/bin/bash
# Set PYTHONPATH to ensure our modules are found
#SBATCH --job-name=main-bertTopic        # Job name
#SBATCH --output=eCallsAgent/output/log_files/main_%j.out      # Standard output and error log
#SBATCH --error=eCallsAgent/output/log_files/main_%j.err       # Separate file for error logs
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks-per-node=1          # One task per node
#SBATCH --cpus-per-task=16          # Number of CPU cores per task
#SBATCH --gres=gpu:4                # Request 4 GPUs
#SBATCH --partition=qgpu72          # Use qgpu72 partition (nodes with 4 GPUs)
#SBATCH --qos=gpu                   # Required QOS for GPU partitions
#SBATCH --time=72:00:00             # Set time limit to 72 hours
#SBATCH --mail-type=BEGIN,END,FAIL  # Notifications for job begin, end, and failure
#SBATCH --mail-user=ZXiao@walton.uark.edu  # Your email address


echo "=== Starting job at $(date) ==="

# 1. Set CUDA environment variables first
export NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1
export CUDA_HOME=/share/apps/mf/cuda/11.7
export PATH=/usr/bin:$PATH
export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH  # System libraries first
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 2. Clear and load modules
module purge
module load os/el7
module load gcc/11.2.1
module load cuda/11.7
module load python/miniforge-24.3.0

# Save module environment to a file that Python can read
module list > /tmp/loaded_modules.txt

# 3. Source conda and create fresh environment
source /home/zichengx/miniconda3/etc/profile.d/conda.sh
# conda deactivate
# conda env remove -n bertopic_env -y
# conda create -n bertopic_env python=3.9 -y
conda activate bertopic_env

# # 4. Install CUDA packages in specific order
# conda install -c conda-forge cudatoolkit=11.7 -y
# conda install -c rapidsai -c nvidia -c conda-forge \
#     rapids=23.08 \
#     cusolver=11.7 \
#     cuml \
#     cudf \
#     cupy \
#     rmm \
#     dask \
#     dask-cuda \
#     -y

# # 5. Install scikit-image for plotly
# conda install -c conda-forge scikit-image -y

# 6. Set final library paths & Set environment variables for Python
export PYTHONPATH=$PYTHONPATH:/scrfs/storage/zichengx/home/Research/AIphaBiz/poetry-demo
export CUDA_MODULE_LOADED=1  # Flag for Python to know CUDA is already loaded

# Verify CUDA
echo "=== Verifying CUDA setup ==="
nvidia-smi || { echo "ERROR: nvidia-smi failed. Exiting."; exit 1; }
which nvidia-smi
nvidia-smi || echo "nvidia-smi failed"
# Verify environment variable is set
echo "NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=$NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY"


# # Install Poetry
# pip install poetry

# Navigate to project directory
cd /scrfs/storage/zichengx/home/Research/AIphaBiz/poetry-demo

# # Clean install with Poetry
# rm -f poetry.lock
# poetry lock
# poetry install

# # Install kaleido directly to ensure it's available
# pip install kaleido

# # Download required NLTK and spaCy data
# python -c "import nltk; nltk.download('wordnet')"
# python -m spacy download en_core_web_sm

# Verification
echo "=== Final Verification ==="
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import nltk, spacy, openai, cuml, cupy, dask, dask_cuda; print('All core packages installed')"

# conda install -c rapidsai -c conda-forge cubinlinker ptxcompiler

# Run the main script
# env NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1 poetry run python -m eCallsAgent.main
python -m eCallsAgent.main

echo "=== Job completed at $(date) ==="
