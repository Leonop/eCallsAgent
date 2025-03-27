"""
Utility module for configuring CUDA environment and handling GPU setup.
"""

# Set Numba CUDA compatibility environment variable
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
        tuple: (cuda_ready, device_str, memory_gb)
            - cuda_ready (bool): Whether CUDA is available and working
            - device_str (str): Device string ('cuda:0' or 'cpu')
            - memory_gb (float): Available GPU memory in GB, or 0 for CPU
    """
    try:
        # Set CUDA environment variables if not already set
        cuda_paths = [
            "/share/apps/mf/cuda/11.7",
            "/usr/local/cuda-11.7",
            "/usr/local/cuda",
            "/opt/cuda-11.7",
            "/opt/cuda"
        ]
        
        # Search for cuda environment variables in module system
        try:
            # Source the module command first
            source_cmd = "source /etc/profile.d/modules.sh"
            # Then run module avail
            module_cmd = "module avail cuda"
            result = subprocess.run(f"{source_cmd} && {module_cmd}", 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE, 
                                 text=True,
                                 shell=True,
                                 executable='/bin/bash')
            
            if result.returncode == 0 and "cuda/11.7" in result.stdout:
                # Load CUDA 11.7
                load_cmd = "module load cuda/11.7"
                subprocess.run(f"{source_cmd} && {load_cmd}", 
                             shell=True,
                             executable='/bin/bash',
                             check=True)
                logger.info("Loaded cuda/11.7 module")
            elif result.returncode == 0 and "cuda/11" in result.stdout:
                # Find best available CUDA 11.x module
                cuda_modules = [line.strip() for line in result.stdout.split('\n') if 'cuda/11' in line]
                if cuda_modules:
                    best_module = sorted(cuda_modules, reverse=True)[0]
                    load_cmd = f"module load {best_module}"
                    subprocess.run(f"{source_cmd} && {load_cmd}", 
                                 shell=True,
                                 executable='/bin/bash',
                                 check=True)
                    logger.info(f"Loaded {best_module} module")
        except Exception as e:
            logger.warning(f"Error when trying to load CUDA module: {e}")
            logger.warning("Continuing with manual CUDA path setup...")
        
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
            # Get number of available GPUs
            num_gpus = torch.cuda.device_count()
            logger.info(f"Found {num_gpus} GPU(s)")
            
            # Try each GPU until we find one that works
            for gpu_id in range(num_gpus):
                try:
                    # Set current device
                    torch.cuda.set_device(gpu_id)
                    device = f"cuda:{gpu_id}"
                    device_name = torch.cuda.get_device_name(gpu_id)
                    cuda_version = torch.version.cuda
                    
                    # Log GPU info
                    logger.info(f"Trying GPU {gpu_id}: {device_name}")
                    logger.info(f"CUDA Version: {cuda_version}")
                    
                    # Validate CUDA version - we need 11.x for compatibility
                    if cuda_version and cuda_version.startswith('11.'):
                        logger.info(f"Found compatible CUDA version: {cuda_version}")
                    elif cuda_version:
                        logger.warning(f"CUDA version {cuda_version} detected. This may not be fully compatible with the required CUDA 11.7.")
                    
                    # Test CUDA with a simple tensor operation
                    try:
                        # Explicitly set current device to gpu_id
                        torch.cuda.set_device(gpu_id)
                        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
                        y = x * 2
                        
                        # Calculate and log GPU memory
                        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
                        total_memory_gb = total_memory / (1024**3)
                        logger.info(f"GPU {gpu_id} memory available: {total_memory_gb:.2f} GB")
                        logger.info(f"CUDA tensor operation successful on GPU {gpu_id}")
                        
                        # Try to fix common CUDA errors
                        try:
                            # Tell PyTorch to use deterministic algorithms when possible
                            torch.backends.cudnn.deterministic = True
                            # Use the fastest convolution algorithms if not deterministic
                            torch.backends.cudnn.benchmark = True
                            logger.info("Set CUDNN optimization parameters")
                        except Exception as e:
                            logger.warning(f"Could not set CUDNN parameters: {e}")
                        
                        return True, device, total_memory_gb
                    except Exception as e:
                        logger.warning(f"GPU {gpu_id} test failed: {e}")
                        continue
                except Exception as e:
                    logger.warning(f"Error setting up GPU {gpu_id}: {e}")
                    continue
            
            # If we get here, no GPU worked
            logger.warning("No working GPU found, falling back to CPU")
            return False, "cpu", 0.0
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
        return False, "cpu", 0.0
    
    except Exception as e:
        logger.error(f"Error setting up CUDA: {e}")
        logger.error(traceback.format_exc())
        return False, "cpu", 0.0

def check_cuml_availability():
    """
    Check if cuML is available and compatible with the current environment.
    
    Returns:
        bool: True if cuML is available and working, False otherwise
    """
    try:
        # First check if we're on CUDA
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available, so cuML will not be available")
            return False
        
        # Try to import and initialize cuML
        import cuml
        version = cuml.__version__
        logger.info(f"cuML version: {version}")
        
        # Run a very small test to verify cuML is working
        try:
            from cuml.datasets import make_blobs
            X, y = make_blobs(n_samples=10, n_features=5, centers=2, random_state=0)
            logger.info("Successfully created cuML test dataset")
            return True
        except Exception as e:
            logger.error(f"cuML test failed: {e}")
            logger.error(traceback.format_exc())
            return False
    except ImportError:
        logger.warning("cuML is not installed or cannot be imported")
        return False
    except Exception as e:
        logger.error(f"Error checking cuML availability: {e}")
        logger.error(traceback.format_exc())
        return False
