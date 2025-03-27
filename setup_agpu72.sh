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

# Create log_files directory if it doesn't exist
mkdir -p log_files

# Function to check command status
check_status() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed"
        exit 1
    else
        echo "✓ $1 completed successfully"
    fi
}

echo "=== Step 1: Finding Conda Installation ==="
# Let's try multiple possible locations for conda
CONDA_LOCATIONS=(
    "/scrfs/storage/zichengx/home/miniconda3"
    "/home/zichengx/miniconda3"
    "/scrfs/opt8/python/miniforge-24.3.0"
    "/scrfs/opt8/python/miniforge-24.11.3"
    "/opt/conda"
    "/share/apps/anaconda3"
    "/share/apps/miniconda3"
)

CONDA_FOUND=false
for conda_path in "${CONDA_LOCATIONS[@]}"; do
    if [ -d "$conda_path" ]; then
        if [ -f "$conda_path/etc/profile.d/conda.sh" ]; then
            echo "Found conda at: $conda_path"
            source "$conda_path/etc/profile.d/conda.sh"
            CONDA_FOUND=true
            break
        fi
    fi
done

# If conda not found in predefined locations, search for it
if [ "$CONDA_FOUND" = false ]; then
    echo "Searching for conda installation..."
    FOUND_CONDA=$(find /scrfs /home /opt /share -name conda.sh -type f 2>/dev/null | head -n 1)
    
    if [ -n "$FOUND_CONDA" ]; then
        echo "Found conda at: $FOUND_CONDA"
        source "$FOUND_CONDA"
        CONDA_FOUND=true
    fi
fi

# If still not found, check if it's already in PATH
if [ "$CONDA_FOUND" = false ] && command -v conda &>/dev/null; then
    echo "Conda is already in PATH: $(which conda)"
    CONDA_FOUND=true
fi

# Final check if conda is available
if [ "$CONDA_FOUND" = false ]; then
    echo "ERROR: Could not find conda installation."
    echo "Please install miniconda3 manually in your home directory using:"
    echo "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3"
    exit 1
fi

# Print conda info
conda info
check_status "Initializing conda"

echo "=== Step 2: Set Up CUDA Environment ==="
# Skip module loading since it's causing errors
echo "Skipping module loading and setting up CUDA environment manually"

# Find CUDA installation
CUDA_LOCATIONS=(
    "/usr/local/cuda-11.7"
    "/usr/local/cuda"
    "/opt/cuda-11.7"
    "/opt/cuda"
)

CUDA_FOUND=false
for cuda_path in "${CUDA_LOCATIONS[@]}"; do
    if [ -d "$cuda_path" ]; then
        echo "Found CUDA at: $cuda_path"
        export CUDA_HOME=$cuda_path
        export PATH=$CUDA_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
        CUDA_FOUND=true
        break
    fi
done

if [ "$CUDA_FOUND" = false ]; then
    echo "WARNING: Could not find CUDA installation."
    # Try to find CUDA dynamically
    if command -v nvidia-smi &>/dev/null; then
        echo "NVIDIA SMI found, attempting to locate CUDA path"
        # Extract CUDA version from nvidia-smi
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1,2 | head -n 1)
        echo "Detected CUDA version: $CUDA_VERSION"
        
        # Look for common CUDA paths based on detected version
        potential_paths=(
            "/usr/local/cuda-$CUDA_VERSION"
            "/usr/local/cuda"
            "/opt/cuda-$CUDA_VERSION"
            "/opt/cuda"
        )
        
        for path in "${potential_paths[@]}"; do
            if [ -d "$path" ]; then
                export CUDA_HOME=$path
                export PATH=$CUDA_HOME/bin:$PATH
                export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
                echo "Found CUDA at: $CUDA_HOME"
                CUDA_FOUND=true
                break
            fi
        done
    fi
fi

# Even if CUDA wasn't found in standard locations, we can continue
if [ "$CUDA_FOUND" = false ]; then
    echo "WARNING: CUDA path not found, but we'll continue anyway."
    # Some installations don't need explicit CUDA_HOME
fi

# Check for basic CUDA tools
if command -v nvcc &>/dev/null; then
    echo "NVCC found: $(which nvcc)"
    nvcc --version
else
    echo "WARNING: NVCC not found in PATH. This may cause issues with CUDA detection."
fi

echo "=== Step 3: Create Base Environment ==="
# Ensure we're not in any conda environment
conda deactivate 2>/dev/null || true
# Remove environment if it exists
conda env remove -n bertopic_env -y 2>/dev/null || true
# Create fresh environment
conda create -n bertopic_env -y python=3.8
check_status "Creating conda environment"
# Activate environment
conda activate bertopic_env
check_status "Activating conda environment"

echo "=== Step 4: Install PyTorch ==="
# Install PyTorch first as it has the most stable CUDA implementation
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117 --force-reinstall --no-deps
pip install --no-deps --force-reinstall numpy==1.23.5
check_status "Installing PyTorch"

echo "=== Step 5: Install RAPIDS cuML for CUDA 11.7 ==="
# First check CUDA version available
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Version:', torch.version.cuda if torch.cuda.is_available() else 'Not available')"

# Use conda to install RAPIDS cuML with exact CUDA version match
conda install -y -c rapidsai -c conda-forge -c nvidia \
    cuml=22.12 cudatoolkit=11.7 python=3.8
check_status "Installing cuML via conda with CUDA 11.7"

echo "=== Step 6: Install Core Dependencies ==="
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
check_status "Installing core dependencies"

echo "=== Step 7: Install Additional Packages ==="
# Clean previous installations to avoid conflicts
pip uninstall -y tokenizers transformers sentence-transformers bertopic

# Install packages one by one with --no-deps to avoid dependency conflicts
pip install --no-cache-dir --no-deps tokenizers==0.13.2
check_status "Installing tokenizers"

pip install --no-cache-dir --no-deps transformers==4.29.0
check_status "Installing transformers"

pip install --no-cache-dir --no-deps sentence-transformers==2.2.2
check_status "Installing sentence-transformers"

pip install --no-cache-dir --no-deps \
    bertopic==0.14.1 \
    gensim==4.2.0 \
    openai==0.28.1 \
    swifter==1.3.4 \
    seaborn==0.12.2
check_status "Installing additional packages"

echo "=== Step 8: Install spaCy via Conda ==="
conda install -y -c conda-forge spacy=3.4.4
check_status "Installing spaCy using conda"

echo "=== Step 9: Download spaCy Model ==="
python -m spacy download en_core_web_sm
check_status "Downloading spaCy model"

echo "=== Step 10: Environment Information ==="
# Print info about installed packages
conda list
echo "=== Python Packages ==="
pip list | grep -E 'torch|cuml|hdbscan|umap|bertopic|sentence-transformers|transformers|tokenizers'

echo "=== Step 11: Test PyTorch CUDA ==="
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'Not available')
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    x = torch.tensor([1.0]).cuda()
    print('CUDA tensor test successful')
else:
    print('CUDA not available. Checking environment:')
    import os
    print('CUDA_HOME:', os.environ.get('CUDA_HOME'))
    print('LD_LIBRARY_PATH:', os.environ.get('LD_LIBRARY_PATH'))
    print('PATH:', os.environ.get('PATH'))
    # Don't exit with error, just warn
    print('WARNING: CUDA not detected but continuing anyway')
"

echo "=== Step 12: Create Environment Instructions File ==="
# Save environment activation instructions to a file
CONDA_PATH=$(which conda | sed 's/\/bin\/conda//')
INSTRUCTIONS_FILE="bertopic_env_instructions.txt"

cat > $INSTRUCTIONS_FILE << EOL
# To activate the bertopic_env environment, use:
source ${CONDA_PATH}/etc/profile.d/conda.sh
conda activate bertopic_env

# Set CUDA environment variables:
export CUDA_HOME=${CUDA_HOME}
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
EOL

echo "=== Setup Complete: Environment 'bertopic_env' is Ready ==="
echo "Instructions for activating this environment have been saved to: $INSTRUCTIONS_FILE"
echo "Use the following in your job scripts:"
cat $INSTRUCTIONS_FILE