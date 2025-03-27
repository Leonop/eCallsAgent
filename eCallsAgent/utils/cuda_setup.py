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
        tuple: (cuda_ready, device, memory_gb)
            - cuda_ready (bool): Whether CUDA is available and working
            - device (str): Device to use ('cuda:0' or 'cpu')
            - memory_gb (float): Available GPU memory in GB, or 0 for CPU
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
                return True, device, total_memory_gb
            except Exception as e:
                logger.error(f"CUDA tensor operation failed: {e}")
                logger.error(traceback.format_exc())
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
    try:
        import cuml
        # Verify cuml is working by accessing its version
        version = cuml.__version__
        logger.info(f"cuML version: {version}")
        return True
    except ImportError:
        return False
