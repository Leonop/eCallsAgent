"""
Tests for the EmbeddingGenerator class.
"""

import os
import numpy as np
import pytest
import torch
from eCallsAgent.core.embedding_generator import EmbeddingGenerator
import eCallsAgent.config.global_options as gl

@pytest.fixture
def embedding_generator():
    """Create an EmbeddingGenerator instance for testing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return EmbeddingGenerator(device)

@pytest.fixture
def sample_docs():
    """Create sample documents for testing."""
    return [
        "This is the first test document.",
        "Here is another document for testing.",
        "A third document with different content."
    ]

def test_initialization(embedding_generator):
    """Test EmbeddingGenerator initialization."""
    assert embedding_generator is not None
    assert embedding_generator.model is not None
    assert embedding_generator.base_batch_size > 0

def test_calculate_optimal_batch_size(embedding_generator):
    """Test optimal batch size calculation."""
    batch_size = embedding_generator._calculate_optimal_batch_size(gl.EMBEDDING_DIM)  # Common embedding dimension
    assert batch_size > 0
    assert isinstance(batch_size, int)

def test_generate_embeddings(embedding_generator, sample_docs):
    """Test embedding generation."""
    embeddings = embedding_generator.generate_embeddings(sample_docs)
    
    # Check embeddings shape and type
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(sample_docs)
    assert embeddings.shape[1] > 0  # Should have some embedding dimensions
    
    # Check if embeddings are normalized
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

def test_save_and_load_embeddings(embedding_generator, sample_docs):
    """Test saving and loading embeddings."""
    # Generate and save embeddings
    embeddings = embedding_generator.generate_embeddings(sample_docs)
    embedding_generator.save_embeddings(embeddings, gl.YEAR_START, gl.YEAR_END)
    
    # Load embeddings
    loaded_embeddings = embedding_generator.load_embeddings(gl.YEAR_START, gl.YEAR_END)
    
    # Compare original and loaded embeddings
    np.testing.assert_array_almost_equal(embeddings, loaded_embeddings)
    
    # Clean up temporary files
    if os.path.exists(gl.TEMP_EMBEDDINGS):
        os.remove(gl.TEMP_EMBEDDINGS)

def test_error_handling(embedding_generator):
    """Test error handling for invalid inputs."""
    with pytest.raises(Exception):
        embedding_generator.generate_embeddings([])  # Empty list should raise error
    
    with pytest.raises(Exception):
        embedding_generator.generate_embeddings([None, None])  # Invalid documents
    
    with pytest.raises(FileNotFoundError):
        embedding_generator.load_embeddings(9999, 9999)  # Non-existent year range 