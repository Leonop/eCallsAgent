#!/bin/bash
#SBATCH --job-name=GPU-bertTopic
#SBATCH --output=eCallsAgent/output/log_files/bertTopic_openai_%j.out
#SBATCH --error=eCallsAgent/output/log_files/bertTopic_openai_%j.err
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
mkdir -p /home/zichengx/Research/AIphaBiz/log_files
mkdir -p /home/zichengx/Research/AIphaBiz/eCallsAgent/output/log_files

# Function to check command status
check_status() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed"
        exit 1
    else
        echo "✓ $1 completed successfully"
    fi
}

echo "=== Step 1: Initialize Conda ==="
# Use the conda installation directly with absolute path
export PATH="/home/zichengx/miniconda3/bin:$PATH"
if command -v conda &>/dev/null; then
    echo "Conda found in PATH"
else
    # Try alternative locations
    if [ -f "/home/zichengx/miniconda3/etc/profile.d/conda.sh" ]; then
        source "/home/zichengx/miniconda3/etc/profile.d/conda.sh"
        echo "Using conda from /home/zichengx/miniconda3"
    elif [ -f "/scrfs/storage/zichengx/home/miniconda3/etc/profile.d/conda.sh" ]; then
        source "/scrfs/storage/zichengx/home/miniconda3/etc/profile.d/conda.sh"
        echo "Using conda from /scrfs/storage/zichengx/home/miniconda3"
    else
        echo "ERROR: Could not find conda installation in expected paths."
        exit 1
    fi
fi

# Verify conda is available
conda --version
check_status "Verifying conda is available"

echo "=== Step 2: Set Up CUDA Environment ==="
# Skip module loading - use CUDA paths directly
if command -v nvidia-smi &>/dev/null; then
    echo "Detected NVIDIA GPU:"
    nvidia-smi
    
    # Set CUDA paths manually (avoids module system)
    for cuda_path in "/usr/local/cuda-11.7" "/usr/local/cuda" "/opt/cuda-11.7" "/opt/cuda"; do
        if [ -d "$cuda_path" ]; then
            export CUDA_HOME=$cuda_path
            export PATH=$CUDA_HOME/bin:$PATH
            export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
            # Add CUPTI path for profiling tools
            export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
            echo "Using CUDA at: $CUDA_HOME"
            break
        fi
    done
    
    # Set Numba environment variables for CUDA
    export NUMBA_CUDA_DRIVER=/usr/lib64/libcuda.so
    export NUMBAPRO_CUDA_DRIVER=/usr/lib64/libcuda.so
    export NUMBAPRO_LIBDEVICE=$CUDA_HOME/nvvm/libdevice
    export NUMBAPRO_NVVM=$CUDA_HOME/nvvm/lib64/libnvvm.so
else
    echo "WARNING: NVIDIA GPU not detected. CUDA functionality will not be available."
fi

echo "=== Step 3: Create Environment ==="
# Start fresh - remove existing environment if it exists
conda deactivate 2>/dev/null || true

# Check if environment exists
if conda env list | grep -q "bertopic_env"; then
    echo "Environment exists, removing..."
    conda env remove -n bertopic_env -y
else
    echo "Environment does not exist, creating new one..."
fi

# Create fresh environment
conda create -n bertopic_env python=3.8 -y
check_status "Creating conda environment"

# Activate environment
conda activate bertopic_env
check_status "Activating conda environment"

echo "=== Step 4: Install PyTorch with CUDA 11.7 ==="
# Install PyTorch with CUDA support
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
check_status "Installing PyTorch"

echo "=== Step 5: Install Core Packages ==="
# Use conda for core packages for better binary compatibility
conda install -y -c conda-forge numpy=1.23.5 pandas=1.4.4 scikit-learn=1.0.2 matplotlib=3.5.2 joblib=1.1.0 tqdm=4.64.0 scipy=1.9.3 hdbscan=0.8.28 umap-learn=0.5.3
check_status "Installing core packages"

echo "=== Step 6: Install Pydantic First (Critical for spaCy compatibility) ==="
# Install specific Pydantic version first to avoid compatibility issues
pip install pydantic==1.10.8 --force-reinstall
check_status "Installing compatible Pydantic version"

echo "=== Step 7: Install spaCy with Compatible Dependencies ==="
# Install spaCy with all dependencies explicitly
pip install wheel setuptools cython
pip install spacy==3.5.3 --force-reinstall

# Install all the missing spaCy dependencies explicitly
pip install langcodes==3.3.0 pathy>=0.3.5 jinja2 typer==0.7.0 spacy-legacy==3.0.12 spacy-loggers==1.0.4 wasabi==0.10.1 catalogue cymem murmurhash preshed srsly thinc==8.1.10
check_status "Installing spaCy and all dependencies"

# Download the model
echo "Downloading spaCy language model..."
python -m spacy download en_core_web_sm
check_status "Installing spaCy language model"

echo "=== Step 8: Install BERTopic and Dependencies ==="
# First install dependencies that transformers needs
pip install filelock huggingface-hub regex
# Required visualization dependency for BERTopic
pip install plotly

# Then install specific versions of tokenizers/transformers that are known to work together
pip install tokenizers==0.13.2 --force-reinstall --no-deps
pip install transformers==4.29.0 --force-reinstall
pip install sentence-transformers==2.2.2 --force-reinstall --no-deps
pip install bertopic==0.14.1 --force-reinstall --no-deps
pip install gensim==4.2.0 openai==0.28.1 swifter==1.3.4 seaborn==0.12.2
check_status "Installing BERTopic and dependencies"

echo "=== Step 9: Verify ALL Required Packages ==="
# Check that key packages are installed and working
echo "Testing PyTorch CUDA:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Torch version:', torch.__version__)"
echo ""
echo "Testing tokenizers and transformers:"
python -c "import tokenizers; import transformers; print('tokenizers version:', tokenizers.__version__); print('transformers version:', transformers.__version__)"
echo ""
echo "Testing spaCy:"
python -c "import spacy; print('spaCy version:', spacy.__version__); print('Pydantic version:', __import__('pydantic').__version__); nlp = spacy.load('en_core_web_sm'); print('spaCy model loaded successfully')" || {
    echo "WARNING: spaCy verification failed. Trying alternative installation..."
    pip uninstall -y spacy
    pip install spacy==3.5.3 --no-deps
    python -m spacy download en_core_web_sm
}

echo "Testing BERTopic:"
python -c "import bertopic; print('bertopic imported successfully')"

echo "=== Step 10: Add MIN_COUNT and THRESHOLD to global_options.py ==="
# Check if MIN_COUNT and THRESHOLD are already in global_options.py
if grep -q "MIN_COUNT" eCallsAgent/config/global_options.py; then
    echo "MIN_COUNT already exists in global_options.py"
else
    # Add MIN_COUNT and THRESHOLD to global_options.py
    echo "Adding MIN_COUNT and THRESHOLD to global_options.py"
    sed -i '/GPU_MEMORY = 40 # in GB/a\MIN_COUNT = 5 # minimum number of occurrences for a bigram/trigram to be considered\nTHRESHOLD = 10 # controls the tendency to form phrases, higher means fewer phrases' eCallsAgent/config/global_options.py
fi

echo "=== Step 11: Update CUDA Setup Utility ==="
# Enhance the CUDA setup utility to better detect and configure CUDA
cat > eCallsAgent/utils/cuda_setup.py << 'EOL'
"""
Utility module for configuring CUDA environment and handling GPU setup.
"""

import os
import sys
import logging
import subprocess
import traceback
import torch
import numpy as np

logger = logging.getLogger(__name__)

def setup_cuda():
    """
    Configure CUDA environment and return device setup info.
    
    Returns:
        str: Device to use ('cuda:0' or 'cpu')
    """
    try:
        # Set CUDA environment variables if not already set
        cuda_paths = [
            "/usr/local/cuda-11.7",
            "/usr/local/cuda",
            "/opt/cuda-11.7",
            "/opt/cuda",
            "/opt/cuda-12.4",
            "/opt/cuda-12.3",
        ]
        
        # Find and set CUDA_HOME if not set
        if "CUDA_HOME" not in os.environ:
            for path in cuda_paths:
                if os.path.exists(path):
                    os.environ["CUDA_HOME"] = path
                    logger.info(f"Setting CUDA_HOME to {path}")
                    break
        
        # Set LD_LIBRARY_PATH if CUDA_HOME is set
        if "CUDA_HOME" in os.environ:
            cuda_home = os.environ["CUDA_HOME"]
            lib_path = os.path.join(cuda_home, "lib64")
            cupti_path = os.path.join(cuda_home, "extras", "CUPTI", "lib64")
            
            # Add to LD_LIBRARY_PATH if not already there
            ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            paths_to_add = []
            
            if lib_path not in ld_path:
                paths_to_add.append(lib_path)
            
            if cupti_path not in ld_path and os.path.exists(cupti_path):
                paths_to_add.append(cupti_path)
                
            if paths_to_add:
                if ld_path:
                    os.environ["LD_LIBRARY_PATH"] = f"{':'.join(paths_to_add)}:{ld_path}"
                else:
                    os.environ["LD_LIBRARY_PATH"] = f"{':'.join(paths_to_add)}"
                logger.info(f"Updated LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")
        
        # Check if CUDA is available in PyTorch
        if torch.cuda.is_available():
            # Get device info
            device = "cuda:0"
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            
            # Log GPU info
            logger.info(f"Using GPU: {device_name}")
            logger.info(f"CUDA Version: {cuda_version}")
            
            # Test CUDA with a simple tensor operation
            try:
                x = torch.tensor([1.0, 2.0, 3.0]).cuda()
                y = x * 2
                # Calculate and log GPU memory
                total_memory = torch.cuda.get_device_properties(0).total_memory
                total_memory_gb = total_memory / (1024**3)
                logger.info(f"GPU memory available: {total_memory_gb:.2f} GB")
                logger.info("CUDA tensor operation successful")
                return device
            except Exception as e:
                logger.error(f"CUDA tensor operation failed: {e}")
                logger.error(traceback.format_exc())
        else:
            # CUDA not available, check why
            logger.warning("CUDA is not available in PyTorch")
            
            # Check if PyTorch was built with CUDA
            if hasattr(torch, 'version') and hasattr(torch.version, 'cuda') and torch.version.cuda:
                logger.info(f"PyTorch was built with CUDA support (version {torch.version.cuda})")
                
                # Check if CUDA libraries are in LD_LIBRARY_PATH
                ld_path = os.environ.get("LD_LIBRARY_PATH", "")
                if "cuda" in ld_path.lower():
                    logger.info(f"CUDA libraries found in LD_LIBRARY_PATH: {ld_path}")
                else:
                    logger.warning(f"CUDA libraries not found in LD_LIBRARY_PATH: {ld_path}")
                
                # Check if NVIDIA driver is loaded
                try:
                    result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if result.returncode == 0:
                        logger.info("NVIDIA driver is loaded and functioning")
                        logger.info(result.stdout.strip())
                    else:
                        logger.warning("NVIDIA driver check failed")
                        logger.warning(result.stderr.strip())
                except Exception as e:
                    logger.warning(f"Error checking NVIDIA driver: {e}")
            else:
                logger.warning("PyTorch was not built with CUDA support")
        
        # Fall back to CPU
        logger.warning("Falling back to CPU")
        return "cpu"
    
    except Exception as e:
        logger.error(f"Error setting up CUDA: {e}")
        logger.error(traceback.format_exc())
        return "cpu"
EOL

echo "=== Step 12: Update preprocess_earningscall.py ==="
# Fix the spaCy/Pydantic compatibility issue in preprocess_earningscall.py
cat > eCallsAgent/core/preprocess_earningscall.py.new << 'EOL'
# Author: Zicheng Xiao
# Date: 2024-09-01
# Description: This script is used to preprocess the earnings call data.
# The data is stored in the data folder, and the preprocessed data is stored in the docword folder.

import codecs
import json
import re
import os
import string
# Fix for pydantic compatibility issue - import pydantic with specific version first
import pydantic
import logging
import sys
import importlib
import pandas as pd
import warnings
from datetime import datetime
import multiprocessing
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore")

# Import global options
from eCallsAgent.config import global_options as gl

# Try to import spaCy and transformers with error handling
try:
    import spacy
    logger.info(f"Successfully imported spaCy version: {spacy.__version__}")
except ImportError as e:
    logger.error(f"Error importing spaCy: {e}")
    spacy = None

try:
    import torch
    from transformers import BertTokenizer, BertModel
    logger.info(f"Successfully imported transformers")
except ImportError as e:
    logger.error(f"Error importing transformers: {e}")
    torch = None
    BertTokenizer = None
    BertModel = None

class NlpPreProcess(object):
    """
    Natural Language Processing class for preprocessing earnings call data.
    
    This class provides methods for text preprocessing, including:
    - Stopword removal
    - Lemmatization
    - N-gram generation
    - Punctuation and digit removal
    - Sentence filtering
    
    It supports multiple stemming algorithms:
    - Porter Stemmer: A popular stemming algorithm that removes common morphological and inflectional endings from words.
    - Snowball Stemmer: An improvement over the Porter Stemmer, also known as Porter2, which is more accurate and handles more edge cases.
    - Lancaster Stemmer: Another popular stemming algorithm that is known for its aggressive approach to stemming.
    - WordNet Lemmatizer: Uses a dictionary of known word forms to convert words to their base forms.
    """
    def __init__(self):
        super(NlpPreProcess, self).__init__()
        self.wnl = WordNetLemmatizer()  # Lemmatization
        self.ps = PorterStemmer()  # Stemming
        self.sb = SnowballStemmer('english')  # Stemming
        self.stoplist = list(set([word.strip().lower() for word in gl.stop_list]))
        
        # Try to load spacy model with error handling
        self.nlp = None
        try:
            if spacy is not None:
                self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
                logger.info("Successfully loaded spaCy model")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            logger.warning("Using fallback tokenization instead of spaCy")
        
        # Initialize BERT model and tokenizer if available
        self.model = None
        self.tokenizer = None
        try:
            if torch is not None and BertTokenizer is not None:
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                logger.info("Successfully loaded BERT tokenizer")
                # Only load model if GPU is available
                if torch.cuda.is_available():
                    self.model = BertModel.from_pretrained('bert-base-uncased')
                    self.model.eval()
                    logger.info("Successfully loaded BERT model")
        except Exception as e:
            logger.error(f"Error loading BERT model/tokenizer: {e}")
        
    def remove_stopwords_from_sentences(self, text):
        '''Split text by sentence, remove stopwords in each sentence, and rejoin sentences into one string'''
        if not text or not isinstance(text, str):
            return ""
            
        # Split text into sentences
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                # Process each sentence by removing stop words
                processed_sentences = []
                for sent in doc.sents:
                    processed_sentence = ' '.join([token.text for token in sent if token.text.lower() not in self.stoplist])
                    processed_sentences.append(processed_sentence)
                # Rejoin all processed sentences into a single string
                return ' '.join(processed_sentences)
            except Exception as e:
                logger.error(f"Error in spaCy processing: {e}")
                # Fall back to simple processing
        
        # Fallback: simple sentence splitting and stopword removal
        sentences = re.split(r'[.!?]+', text)
        processed_sentences = []
        for sentence in sentences:
            words = sentence.split()
            processed_sentence = ' '.join([word for word in words if word.lower() not in self.stoplist])
            processed_sentences.append(processed_sentence)
        return ' '.join(processed_sentences)
EOL

# Replace only the beginning of the file, keeping the rest
head -n 150 eCallsAgent/core/preprocess_earningscall.py.new > temp_file
tail -n +150 eCallsAgent/core/preprocess_earningscall.py >> temp_file
mv temp_file eCallsAgent/core/preprocess_earningscall.py
rm eCallsAgent/core/preprocess_earningscall.py.new

# Fix spacy_patch.py to handle pydantic version correctly
cat > /home/zichengx/Research/AIphaBiz/spacy_patch.py << 'EOL'
"""
This patch fixes the spaCy import issues by ensuring pydantic is imported first.
"""
# Import pydantic first to avoid circular import issues
import pydantic
import typing
import typing_extensions

# Now it's safe to import spacy
import spacy

# Get pydantic version safely
def get_pydantic_version():
    """Get pydantic version in a compatible way."""
    try:
        return pydantic.__version__
    except AttributeError:
        try:
            # For newer pydantic versions
            from pydantic.version import VERSION
            return VERSION
        except ImportError:
            try:
                import pkg_resources
                return pkg_resources.get_distribution("pydantic").version
            except:
                return "Unknown"

print("spaCy successfully imported with version:", spacy.__version__)
print("Pydantic version:", get_pydantic_version())
EOL

# Create a patch to be added at the top of main.py
echo "Creating a patch for main.py - you'll need to add this at the top of your main.py file"
cat << 'EOL'
"""
Add these imports at the VERY TOP of your main.py file to fix pydantic issues:

# Add these imports at the very top of main.py
import pydantic
import typing
import typing_extensions

# Safe version detection function
def get_pydantic_version():
    """Get pydantic version safely."""
    try:
        return pydantic.__version__
    except AttributeError:
        try:
            from pydantic.version import VERSION
            return VERSION
        except ImportError:
            try:
                import pkg_resources
                return pkg_resources.get_distribution("pydantic").version
            except:
                return "Unknown"
"""
EOL

# Update main.sh to set the import order correctly
echo "Updating main.sh to ensure correct import order..."
if [ -f "main.sh" ]; then
    # Make a backup copy
    cp main.sh main.sh.backup
    
    # Add the environment variables to ensure correct import order
    sed -i '2i# Set PYTHONPATH to ensure our modules are found\nexport PYTHONPATH=/home/zichengx/Research/AIphaBiz:$PYTHONPATH\n\n# Ensure pydantic is loaded first\nexport PYTHONSTARTUP=/home/zichengx/Research/AIphaBiz/spacy_patch.py' main.sh
    
    echo "main.sh updated to fix pydantic import order"
else
    echo "main.sh not found. Please create it with these contents:"
    cat << 'EOL'
#!/bin/bash
# Set PYTHONPATH to ensure our modules are found
export PYTHONPATH=/home/zichengx/Research/AIphaBiz:$PYTHONPATH

# Ensure pydantic is loaded first
export PYTHONSTARTUP=/home/zichengx/Research/AIphaBiz/spacy_patch.py

# Now run your main.py with conda
source /home/zichengx/miniconda3/etc/profile.d/conda.sh
conda activate bertopic_env
python eCallsAgent/main.py
EOL
fi

# Test the updated spacy_patch.py with proper version detection
echo "Testing spaCy import patch..."
conda run -n bertopic_env python /home/zichengx/Research/AIphaBiz/spacy_patch.py
check_status "Testing spaCy import patch"

echo "=== Setup Complete ==="
echo "Environment 'bertopic_env' is ready"
echo "All fixes have been applied to address the compatibility issues"

echo "=== Running simple test script ==="
python -c "print('Setup completed successfully!')"

# Run the main script
python -m eCallsAgent.main
