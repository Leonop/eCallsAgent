"""
Tests for visualization functions.
"""

import pytest
import numpy as np
from plotly.graph_objects import Figure
from eCallsAgent.utils.visualization import (
    create_topic_heatmap,
    create_topic_hierarchy,
    create_intertopic_distance_map
)

class MockTopicModel:
    """Mock BERTopic model for testing."""
    def __init__(self):
        self.topic_similarities_ = np.array([[1.0, 0.5], [0.5, 1.0]])
        self.topic_embeddings_ = np.random.randn(2, 10)  # 2 topics, 10 dimensions
        
    def get_topic_info(self):
        import pandas as pd
        return pd.DataFrame({
            'Topic': [0, 1],
            'Count': [100, 50]
        })
    
    def get_topic(self, topic_id):
        return [('word1', 0.5), ('word2', 0.3), ('word3', 0.2)]

@pytest.fixture
def mock_topic_model():
    """Create a mock topic model for testing."""
    return MockTopicModel()

@pytest.fixture
def custom_labels():
    """Create custom labels for testing."""
    return {0: "Business", 1: "Technology"}

def test_create_topic_heatmap(mock_topic_model, custom_labels):
    """Test topic heatmap creation."""
    fig = create_topic_heatmap(mock_topic_model, custom_labels)
    assert isinstance(fig, Figure)
    assert fig.layout.title.text == "Topic Similarity Heatmap"

def test_create_topic_hierarchy(mock_topic_model, custom_labels):
    """Test topic hierarchy visualization."""
    fig = create_topic_hierarchy(mock_topic_model, custom_labels)
    assert isinstance(fig, Figure)
    assert fig.layout.title.text == "Topic Hierarchy"

def test_create_intertopic_distance_map(mock_topic_model, custom_labels):
    """Test intertopic distance map creation."""
    fig = create_intertopic_distance_map(mock_topic_model, custom_labels)
    assert isinstance(fig, Figure)
    assert fig.layout.title.text == "Intertopic Distance Map"

def test_visualization_dimensions():
    """Test custom dimensions for visualizations."""
    mock_model = MockTopicModel()
    width, height = 800, 600
    
    # Test heatmap dimensions
    fig1 = create_topic_heatmap(mock_model, width=width, height=height)
    assert fig1.layout.width == width
    assert fig1.layout.height == height
    
    # Test hierarchy dimensions
    fig2 = create_topic_hierarchy(mock_model, width=width, height=height)
    assert fig2.layout.width == width
    assert fig2.layout.height == height
    
    # Test distance map dimensions
    fig3 = create_intertopic_distance_map(mock_model, width=width, height=height)
    assert fig3.layout.width == width
    assert fig3.layout.height == height

def test_error_handling():
    """Test error handling in visualization functions."""
    invalid_model = None
    
    with pytest.raises(Exception):
        create_topic_heatmap(invalid_model)
    
    with pytest.raises(Exception):
        create_topic_hierarchy(invalid_model)
    
    with pytest.raises(Exception):
        create_intertopic_distance_map(invalid_model) 