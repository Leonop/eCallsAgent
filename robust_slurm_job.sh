#!/bin/bash
#SBATCH --job-name=bertopic-fixed
#SBATCH --output=eCallsAgent/output/log_files/bertopic_%j.out
#SBATCH --error=eCallsAgent/output/log_files/bertopic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=agpu72
#SBATCH --qos=gpu
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ZXiao@walton.uark.edu

# Create log directories
mkdir -p /home/zichengx/Research/AIphaBiz/eCallsAgent/output/log_files
mkdir -p /home/zichengx/Research/AIphaBiz/log_files

# Print job information
echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Submitted from: $(hostname)"
echo "Current directory: $(pwd)"
echo "Date: $(date)"

# Use absolute paths for everything
MINICONDA_DIR="/home/zichengx/miniconda3"
WORK_DIR="/scrfs/storage/zichengx/home/Research/AIphaBiz"
ENV_NAME="bertopic_env"

# Initialize conda
echo "=== Initializing conda ==="
if [ -f "$MINICONDA_DIR/etc/profile.d/conda.sh" ]; then
    source "$MINICONDA_DIR/etc/profile.d/conda.sh"
    echo "Sourced conda.sh from $MINICONDA_DIR"
else
    echo "ERROR: conda.sh not found at $MINICONDA_DIR/etc/profile.d/conda.sh"
    echo "Searching for alternative conda installations..."
    
    # Look for conda in alternate locations
    for loc in "/scrfs/storage/zichengx/home/miniconda3" "/opt/conda" "$HOME/miniconda3" "$HOME/anaconda3"; do
        if [ -f "$loc/etc/profile.d/conda.sh" ]; then
            source "$loc/etc/profile.d/conda.sh"
            MINICONDA_DIR="$loc"
            echo "Found and sourced conda.sh from $loc"
            break
        fi
    done
    
    # Check if conda was found
    if ! command -v conda &> /dev/null; then
        echo "ERROR: conda command not available. Exiting."
        exit 1
    fi
fi

# Print conda information
echo "Conda version: $(conda --version)"
echo "Conda location: $(which conda)"

# Set up CUDA environment
echo "=== Setting up CUDA environment ==="
for cuda_path in "/usr/local/cuda-11.7" "/usr/local/cuda" "/opt/cuda-11.7" "/opt/cuda"; do
    if [ -d "$cuda_path" ]; then
        export CUDA_HOME="$cuda_path"
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
        echo "Using CUDA from: $CUDA_HOME"
        break
    fi
done

# Check if we have CUDA
if [ -z "$CUDA_HOME" ]; then
    echo "WARNING: CUDA installation not found"
else
    echo "Checking NVIDIA driver..."
    nvidia-smi || echo "WARNING: nvidia-smi failed"
fi

# Activate the conda environment
echo "=== Activating conda environment ==="
conda activate "$ENV_NAME"

# Check if activation was successful
if [[ "$CONDA_PREFIX" != *"$ENV_NAME"* ]]; then
    echo "ERROR: Failed to activate environment $ENV_NAME"
    echo "Available environments:"
    conda env list
    
    echo "Setting up a fresh environment..."
    # Run the fix script
    bash "$WORK_DIR/slurm_fix_env.sh"
    
    # Try activating again
    conda activate "$ENV_NAME"
    if [[ "$CONDA_PREFIX" != *"$ENV_NAME"* ]]; then
        echo "ERROR: Still failed to activate environment. Exiting."
        exit 1
    fi
fi

echo "Successfully activated $ENV_NAME"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Set PYTHONPATH to ensure our modules are found
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

# Ensure pydantic is loaded first via PYTHONSTARTUP
export PYTHONSTARTUP="$WORK_DIR/spacy_patch.py"

# Run additional fix to ensure NLTK is installed
echo "=== Installing NLTK data if needed ==="
python -c "
import nltk
import os
import sys

required_data = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')

if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)
    
for item in required_data:
    try:
        nltk.data.find(f'{item}')
        print(f'NLTK data {item} already downloaded')
    except LookupError:
        print(f'Downloading {item}...')
        nltk.download(item)
"

# Verify critical packages are installed
echo "=== Verifying packages ==="
python -c "
import sys
try:
    import transformers
    import spacy
    import nltk
    import torch
    import bertopic
    print('All required packages are installed')
except ImportError as e:
    print(f'ERROR: Missing package: {e}')
    sys.exit(1)
"

# Check if verification was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Package verification failed. Installing missing packages..."
    pip install transformers==4.29.0 spacy==3.5.3 nltk bertopic torch==1.13.1
    python -m spacy download en_core_web_sm
fi

# Change to working directory
cd "$WORK_DIR"

# Run the actual job
echo "=== Starting main job ==="
python -m eCallsAgent.main

echo "=== Job completed ==="
date
