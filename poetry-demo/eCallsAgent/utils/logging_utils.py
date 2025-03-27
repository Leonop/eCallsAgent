'''
Purpose of Logging Utils:
- Provide a consistent logging mechanism across the project
- Log important events, errors, and system information
- Help with debugging and diagnosing issues
- Provide a record of the project's execution
'''

import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logging(log_dir='logs'):
    """
    Configure logging for the application with both file and console output.
    
    Args:
        log_dir: Directory to store log files
    
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f'bertopic_processing_{timestamp}.log')
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create rotating file handler (max 10MB per file, keep 5 backup files)
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Get the root logger and configure it
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers and add our handlers
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log initial information
    logger.info("="*50)
    logger.info("Logging initialized")
    logger.info(f"Log file: {log_filename}")
    logger.info("="*50)
    
    return logger

def log_gpu_info():
    """Log GPU information if available."""
    try:
        import torch
        logger = logging.getLogger(__name__)
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"Found {gpu_count} GPU(s):")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
                logger.info(f"  GPU {i}: {gpu_name} ({total_memory:.2f} GB)")
        else:
            logger.warning("No GPU available. Using CPU only.")
    except Exception as e:
        logger.error(f"Error getting GPU information: {e}")

def log_system_info():
    """Log system information."""
    import platform
    import psutil
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("System Information:")
        logger.info(f"  OS: {platform.system()} {platform.release()}")
        logger.info(f"  Python: {platform.python_version()}")
        logger.info(f"  CPU Cores: {psutil.cpu_count()}")
        logger.info(f"  Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    except Exception as e:
        logger.error(f"Error getting system information: {e}") 