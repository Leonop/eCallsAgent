"""
Mock visualization module for testing.

This module provides mock implementations of the visualization functions
used in topic_modeler.py, allowing for easier testing without dependencies.
"""

import logging
import unittest.mock as mock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create mock functions with the same signatures as the real ones
def save_barchart(topic_model, top_n_topics, custom_labels, n_topics):
    """Mock implementation of save_barchart."""
    logger.info(f"Mock save_barchart called with {top_n_topics} topics")
    return None

def save_hierarchy(topic_model, custom_labels, n_topics):
    """Mock implementation of save_hierarchy."""
    logger.info(f"Mock save_hierarchy called with {n_topics} topics")
    return None

def save_heatmap(topic_model, custom_labels, n_topics):
    """Mock implementation of save_heatmap."""
    logger.info(f"Mock save_heatmap called with {n_topics} topics")
    return None

def save_distance_map(topic_model, custom_labels, n_topics):
    """Mock implementation of save_distance_map."""
    logger.info(f"Mock save_distance_map called with {n_topics} topics")
    return None

def save_embedding_visualization(embeddings, topics=None, custom_labels=None, sample_size=5000, title="Document Embeddings"):
    """Mock implementation of save_embedding_visualization."""
    logger.info(f"Mock save_embedding_visualization called with {len(embeddings)} embeddings")
    return None

def save_all_visualizations(topic_model, custom_labels, n_topics, top_n_topics=20):
    """Mock implementation of save_all_visualizations."""
    logger.info(f"Mock save_all_visualizations called with {n_topics} topics")
    return None

def get_base_config(width=1600, height=900):
    """Mock implementation of get_base_config."""
    logger.info(f"Mock get_base_config called with width={width}, height={height}")
    return {
        'width': width,
        'height': height,
        'template': 'plotly_white',
    }

# Create a patch context manager for easy mocking
def patch_visualization():
    """Return a patch context manager for all visualization functions."""
    patches = [
        mock.patch('core.visualization.save_barchart', side_effect=save_barchart),
        mock.patch('core.visualization.save_hierarchy', side_effect=save_hierarchy),
        mock.patch('core.visualization.save_heatmap', side_effect=save_heatmap),
        mock.patch('core.visualization.save_distance_map', side_effect=save_distance_map),
        mock.patch('core.visualization.save_embedding_visualization', side_effect=save_embedding_visualization),
        mock.patch('core.visualization.save_all_visualizations', side_effect=save_all_visualizations),
        mock.patch('core.visualization.get_base_config', side_effect=get_base_config),
    ]
    return mock.patch.multiple('', *patches) 