#!/usr/bin/env python
"""
Test script for the EmbeddingGenerator class.
This script tests just the embedding generation functionality
without depending on the full application.
"""
import os
import sys
import logging
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure CUDA compatibility for numba
os.environ['NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY'] = '1'

def test_embedding_generator():
    """Test EmbeddingGenerator initialization and basic functionality."""
    try:
        # Import the module
        from eCallsAgent.core.embedding_generator import EmbeddingGenerator
        from eCallsAgent.utils.cuda_setup import init_sentence_transformer
        
        logger.info("Successfully imported EmbeddingGenerator and cuda_setup")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
        logger.info(f"Using device: {device}")
        
        if cuda_available:
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        
        # Try to initialize a model directly with init_sentence_transformer
        model_name = "all-MiniLM-L6-v2"  # Small model for quick testing
        logger.info(f"Initializing model {model_name} directly with init_sentence_transformer")
        model = init_sentence_transformer(model_name, device)
        logger.info(f"Model loaded successfully. Embedding dimension: {model.get_sentence_embedding_dimension()}")
        
        # Test EmbeddingGenerator class
        logger.info("Initializing EmbeddingGenerator")
        # Use model index 0 as a simpler default
        embedding_generator = EmbeddingGenerator(device=device, model_index=0)
        logger.info(f"EmbeddingGenerator initialized with model: {embedding_generator.model_name}")
        logger.info(f"Embedding dimension: {embedding_generator.embedding_dim}")
        
        # Test embedding generation
        test_docs = [
            "This is a test document.",
            "Another test document with different content.",
            "A third document to ensure batch processing works."
        ]
        
        logger.info(f"Generating embeddings for {len(test_docs)} test documents")
        embeddings = embedding_generator.model.encode(test_docs, device=device)
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        logger.info("Embedding generation test completed successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Check if the required packages are installed and the PYTHONPATH is set correctly")
        return False
    except Exception as e:
        logger.error(f"Error during test: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting embedding generator test")
    
    if 'PYTHONPATH' in os.environ:
        logger.info(f"PYTHONPATH: {os.environ['PYTHONPATH']}")
    else:
        logger.warning("PYTHONPATH environment variable not set")
        
    success = test_embedding_generator()
    
    if success:
        logger.info("Test completed successfully")
        sys.exit(0)
    else:
        logger.error("Test failed")
        sys.exit(1) 