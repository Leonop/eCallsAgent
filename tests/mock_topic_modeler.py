"""
Mock implementation of topic_modeler.py for testing.
"""

import logging
import traceback
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mock functions for testing
save_all_visualizations = MagicMock()
save_embedding_visualization = MagicMock()

def save_figures():
    """Mock implementation of save_figures method."""
    try:
        logger.info("Generating and saving visualizations...")
        
        # Create a mock topic model
        mock_topic_model = MagicMock()
        mock_topic_model.get_topic_info.return_value = pd.DataFrame({
            'Topic': [-1, 0, 1],
            'Count': [5, 10, 8],
            'Name': ['Outlier', 'Topic 1', 'Topic 2']
        })
        mock_topic_model.topics_ = [0, 1, 0, -1, 1]
        
        # Add document_embeddings_ attribute
        mock_topic_model.document_embeddings_ = np.random.rand(10, 2)
        
        # Call the visualization functions
        save_all_visualizations(mock_topic_model, {}, 2)
        
        logger.info("Saving document embeddings visualization...")
        save_embedding_visualization(
            mock_topic_model.document_embeddings_,
            mock_topic_model.topics_,
            custom_labels={},
            sample_size=5000,
            title="Document Embeddings by Topic"
        )
        
        logger.info("All visualizations saved successfully.")
    except Exception as e:
        logger.error(f"Error saving visualizations: {e}")
        logger.error(traceback.format_exc())
        raise 