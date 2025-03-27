#!/bin/bash
#SBATCH --job-name=GPU-bertTopic
#SBATCH --output=log_files/bertTopic_openai_%j.out
#SBATCH --error=log_files/bertTopic_openai_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=agpu72
#SBATCH --qos=gpu
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ZXiao@walton.uark.edu

# Function to check command status
check_status() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed"
        exit 1
    fi
}

# Create log directory if it doesn't exist
mkdir -p log_files

echo "=== Step 1: Clear Environment ==="
module purge
module load os/el7
module load gcc/11.2.1
module load cuda/11.7
module load python/miniforge-24.3.0  # Load the system's conda module
check_status "Loading modules"

echo "=== Step 2: Initialize Conda ==="
if ! command -v conda &>/dev/null; then
    echo "conda command not found. Please ensure the python/miniforge module is loaded."
    exit 1
fi

# Initialize conda using its shell hook
eval "$(conda shell.bash hook)"
check_status "Conda initialization"

echo "=== Step 3: Create Base Environment ==="
conda deactivate 2>/dev/null
conda env remove -n bertopic_env -y 2>/dev/null
conda create -n bertopic_env -y python=3.8
check_status "Creating conda environment"
conda activate bertopic_env
check_status "Activating conda environment"

echo "=== Step 4: Install PyTorch ==="
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117
check_status "Installing PyTorch"

echo "=== Step 5: Install RAPIDS cuML for CUDA 11.7 ==="
# First check CUDA version available
nvcc --version
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Version:', torch.version.cuda)"

# Uninstall any previous cuml attempts that might cause conflicts
pip uninstall -y cuml cuml-cu11

# Use conda to install RAPIDS cuML with exact CUDA version match
conda install -y -c rapidsai-nightly -c conda-forge -c nvidia \
    cuml=23.4 cudatoolkit=11.7 python=3.8
check_status "Installing cuML via conda with CUDA 11.7"

# Diagnostic check for cuML installation
echo "Running cuML installation diagnostic check..."
python -c "
import sys
try:
    import cuml
    print('SUCCESS: cuML imported successfully')
    print('cuML version:', cuml.__version__)
    print('cuML location:', cuml.__file__)
    
    # Test basic CUDA functionality
    try:
        from cuml.cluster import HDBSCAN
        print('SUCCESS: HDBSCAN from cuML can be imported')
    except Exception as e:
        print('ERROR: Could not import HDBSCAN from cuML:', e)
        
    # Check if other RAPIDS components are available
    try:
        import cudf
        print('SUCCESS: cuDF is available, version:', cudf.__version__)
    except ImportError:
        print('INFO: cuDF is not installed (optional)')
except ImportError as e:
    print('ERROR: Failed to import cuML:', e)
    
    # Check CUDA availability via other libraries
    import torch
    print('Python version:', sys.version)
    print('Python executable:', sys.executable)
    print('PyTorch version:', torch.__version__)
    print('CUDA available via PyTorch:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA version (PyTorch):', torch.version.cuda)
        print('CUDA device:', torch.cuda.get_device_name(0))
    
    # List potential cuML packages
    import subprocess
    result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                            capture_output=True, text=True)
    print('\\nInstalled packages containing \"cuml\":')
    for line in result.stdout.split('\\n'):
        if 'cuml' in line.lower():
            print('   ', line)
"

echo "=== Step 6: Install Core Dependencies, HDBSCAN, UMAP, Dask, and SciPy ==="
conda install -y -c conda-forge \
    numpy=1.23.5 \
    pandas=1.4.4 \
    scikit-learn=1.0.2 \
    matplotlib=3.5.2 \
    joblib=1.1.0 \
    tqdm=4.64.0 \
    hdbscan=0.8.28 \
    umap-learn=0.5.3 \
    dask \
    scipy=1.9.3
check_status "Installing core dependencies including HDBSCAN, UMAP, Dask, and SciPy"

echo "=== Step 7: Install Additional Packages (Excluding spaCy) ==="
pip install --no-cache-dir \
    sentence-transformers==2.2.2 \
    bertopic==0.14.1 \
    gensim==4.2.0 \
    openai \
    swifter \
    seaborn \
    huggingface_hub==0.12.1

check_status "Installing additional packages"

# Verify cuML installation again after all package installations
python -c "
import sys
print('Python version:', sys.version)
print('Python executable:', sys.executable)
try:
    import cuml
    print('cuML installation successful!')
    print('cuML version:', cuml.__version__)
    print('cuML path:', cuml.__file__)
except ImportError as e:
    print('cuML import error:', e)
    # Try to get more information
    import subprocess
    result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                            capture_output=True, text=True)
    print('Installed packages containing cuml:')
    for line in result.stdout.split('\\n'):
        if 'cuml' in line.lower():
            print('   ', line)
"

echo "=== Step 8: Install spaCy via Conda ==="
# Installing spaCy via conda avoids build dependency issues from pip.
conda install -y -c conda-forge spacy=3.4.4
check_status "Installing spaCy using conda"

echo "=== Step 9: Download spaCy Model ==="
python -m spacy download en_core_web_sm
check_status "Downloading spaCy model"

echo "=== Step 10: Set Environment Variables ==="
export CUDA_HOME=/usr/local/cuda-11.7
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

echo "=== Step 11: Environment Information ==="
nvidia-smi
nvcc --version
python --version
pip list

echo "=== Step 12: Test PyTorch CUDA ==="
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    x = torch.tensor([1.0]).cuda()
    print('CUDA tensor test successful')
else:
    print('CUDA not available')
    exit(1)
"
check_status "PyTorch CUDA test"

echo "=== Step 13: Test All Dependencies ==="
python -c "
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import swifter
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
import openai
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from matplotlib import pyplot as plt
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech, TextGeneration
from sklearn.decomposition import PCA
import joblib
import collections
import re

try:
    hdb_version = HDBSCAN.__version__
except AttributeError:
    hdb_version = 'unknown'
print('NumPy:', np.__version__)
print('Pandas:', pd.__version__)
print('HDBSCAN:', hdb_version)
print('Gensim:', getattr(__import__(\"gensim\"), '__version__', 'unknown'))
print('Dask:', getattr(__import__(\"dask\"), '__version__', 'unknown'))
print('Swifter:', getattr(__import__(\"swifter\"), '__version__', 'unknown'))
"
check_status "Testing dependencies"

if [ $? -eq 0 ]; then
    echo "=== Step 14: Running Main Script ==="
    cd ~/Research/narrativesBERT/
    python -u BERTtopic_big_data_hpc_v6.py
else
    echo "Environment setup failed"
    exit 1
fi