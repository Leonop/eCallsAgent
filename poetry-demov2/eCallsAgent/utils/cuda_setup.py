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

def setup_cuda(device_id=0):
    """
    Configure CUDA environment and check GPU availability.
    Returns a tuple of (cuda_ready, device_str, available_memory)
    """
    try:
        # First check if CUDA is available through PyTorch
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available in PyTorch")
            return False, "cpu", 0

        # Get device properties
        device_props = torch.cuda.get_device_properties(device_id)
        logger.info(f"Found GPU: {device_props.name}")
        logger.info(f"Compute capability: {device_props.major}.{device_props.minor}")
        
        # Set device
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
        
        # Clear cache and reset GPU
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Try to initialize cupy
        try:
            import cupy
            pool = cupy.cuda.MemoryPool()
            cupy.cuda.set_allocator(pool.malloc)
            logger.info("CuPy initialized successfully")
        except ImportError:
            logger.warning("CuPy not available, continuing without it")
        except Exception as e:
            logger.warning(f"Error initializing CuPy: {e}")
            
        # Get memory info
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        available_memory = total_memory - allocated_memory
        
        logger.info(f"Total GPU memory: {total_memory / 1e9:.2f} GB")
        logger.info(f"Reserved memory: {reserved_memory / 1e9:.2f} GB")
        logger.info(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        logger.info(f"Available memory: {available_memory / 1e9:.2f} GB")
        
        # Test CUDA availability with a small tensor operation
        try:
            test_tensor = torch.ones(1).cuda()
            test_tensor = test_tensor + 1
            del test_tensor
            logger.info("CUDA test tensor operation successful")
        except Exception as e:
            logger.error(f"CUDA test tensor operation failed: {e}")
            return False, "cpu", 0

        logger.info("CUDA setup completed successfully")
        return True, str(device), available_memory
        
    except Exception as e:
        logger.error(f"Error during CUDA setup: {e}")
        logger.error(traceback.format_exc())
        return False, "cpu", 0

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
        try:
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
        # Check if CUDA is actually available first
        if device.lower() != 'cpu' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU mode.")
            device = "cpu"
        
        # For instructor-xl, use explicit configuration
        if "instructor" in model_name.lower():
            from sentence_transformers import models
            word_embedding_model = models.Transformer(model_name)
            
            # Try to move to device safely
            try:
                if device.lower() != 'cpu' and torch.cuda.is_available():
                    word_embedding_model.to(torch.device(device))
                else:
                    word_embedding_model.to(torch.device("cpu"))
                    device = "cpu"
            except Exception as e:
                logger.warning(f"Failed to move model to {device}: {e}")
                word_embedding_model.to(torch.device("cpu"))
                device = "cpu"
                
            # Use the correct pooling configuration for instructor-xl
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode='mean',  # Use simple mean pooling
                pooling_mode_mean_tokens=True,
                pooling_mode_cls_token=False,
                pooling_mode_max_tokens=False,
                pooling_mode_mean_sqrt_len_tokens=False
            )
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            logger.info(f"Successfully loaded model {model_name} with custom pooling configuration on {device}")
            return model
        
        # For other models, try standard initialization
        model = SentenceTransformer(model_name)
        logger.info(f"Successfully loaded model {model_name}")
        
        # Safely move to device
        try:
            if device.lower() != 'cpu':
                model.to(torch.device(device))
                logger.info(f"Model moved to {device}")
            else:
                model.to(torch.device("cpu"))
                logger.info("Model using CPU")
        except Exception as e:
            logger.warning(f"Failed to move model to {device}: {e}")
            logger.warning("Falling back to CPU")
            model.to(torch.device("cpu"))
        
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        # Try simple CPU fallback as last resort
        try:
            logger.warning("Attempting CPU-only mode as last resort")
            model = SentenceTransformer(model_name)
            model.to(torch.device("cpu"))
            logger.info(f"Successfully loaded model {model_name} on CPU")
            return model
        except Exception as cpu_e:
            logger.error(f"CPU fallback also failed: {cpu_e}")
            raise
