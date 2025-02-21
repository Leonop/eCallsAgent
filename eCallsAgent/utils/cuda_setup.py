"""
Utility module for CUDA setup and configuration.
"""

import logging
import torch

logger = logging.getLogger(__name__)

def setup_cuda() -> str:
    """Initialize CUDA if available and return device string."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.empty_cache()
        device = f'cuda:{torch.cuda.current_device()}'
        gpu_props = torch.cuda.get_device_properties(torch.cuda.current_device())
        logger.info(f"Using GPU: {gpu_props.name}")
        logger.info(f"GPU Memory: {gpu_props.total_memory / 1e9:.2f} GB")
        return device
    else:
        logger.warning("CUDA not available, using CPU")
        return "cpu" 