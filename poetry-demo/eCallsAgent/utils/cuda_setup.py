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
import time
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

def setup_cuda():
    """Set up CUDA environment and check GPU availability."""
    # Try to find available GPU
    cuda_ready = False
    device_str = "cpu"
    memory_gb = 0
    
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"Found {gpu_count} GPU(s)")
            
            # Try each GPU with a limited number of retries
            max_retries = 2
            for retry in range(max_retries):
                for i in range(gpu_count):
                    try:
                        device_name = torch.cuda.get_device_name(i)
                        logger.info(f"Trying GPU {i}: {device_name}")
                        
                        # Test GPU with a small operation
                        with torch.cuda.device(i):
                            x = torch.ones(10, device=f'cuda:{i}')
                            y = x + x
                            torch.cuda.synchronize(i)  # Wait for completion
                            
                            # GPU works - get properties
                            props = torch.cuda.get_device_properties(i)
                            memory_gb = props.total_memory / (1024**3)
                            cuda_ready = True
                            device_str = f"cuda:{i}"
                            logger.info(f"Successfully using GPU {i} with {memory_gb:.2f}GB memory")
                            return cuda_ready, device_str, memory_gb
                    except Exception as e:
                        logger.warning(f"GPU {i} unavailable: {str(e)}")
                
                if retry < max_retries - 1:
                    logger.info(f"Retrying GPU initialization (attempt {retry+2}/{max_retries})")
                    time.sleep(1)  # Small delay before retry
            
            logger.warning("All GPUs unavailable. Using CPU instead.")
        else:
            logger.warning("CUDA not available. Using CPU.")
            
        return cuda_ready, device_str, memory_gb
        
    except Exception as e:
        logger.error(f"Error in CUDA setup: {str(e)}")
        return cuda_ready, device_str, memory_gb

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

def init_sentence_transformer(model_name, device="cuda"):
    """Initialize a SentenceTransformer model with fallback for models that have pooling issues.
    
    Args:
        model_name (str): The name of the model to load
        device (str): The device to use for the model
        
    Returns:
        SentenceTransformer: An initialized SentenceTransformer model
    """
 
    try:
        # Try standard initialization first
        model = SentenceTransformer(model_name)
        # Set device after creation
        if device.lower() != 'cpu':
            model.to(torch.device(device))
        logger.info(f"Successfully loaded model {model_name} with standard parameters")
        return model
    except TypeError as e:
        logger.warning(f"Error loading model with default parameters: {e}")
        logger.info("Trying to load with explicit pooling configuration...")
        
        # For instructor-xl and other models that might have pooling mode issues
        from sentence_transformers import models
        word_embedding_model = models.Transformer(model_name)
        # Set device after model creation
        if device.lower() != 'cpu':
            word_embedding_model.to(torch.device(device))
            
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        logger.info(f"Successfully loaded model {model_name} with custom pooling configuration")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise
