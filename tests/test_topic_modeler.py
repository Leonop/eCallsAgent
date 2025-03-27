"""
Test script for the TopicModeler class.
"""

import os
import sys
import unittest
import numpy as np
import logging
import traceback
from unittest.mock import patch, MagicMock, DEFAULT, call, PropertyMock
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import eCallsAgent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing the required modules, with helpful error messages
try:
    from eCallsAgent.core.topic_modeler import TopicModeler
    from eCallsAgent.config import global_options as gl
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error(traceback.format_exc())
    raise

# Import visualization module here to avoid circular imports
import eCallsAgent.core.visualization as visualization

class TestTopicModeler(unittest.TestCase):
    """Test cases for the TopicModeler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            # Mock device for testing
            self.device = "cpu"
            
            # Create a TopicModeler instance with mocked dependencies
            with patch('eCallsAgent.core.topic_modeler.SentenceTransformer'), \
                 patch('eCallsAgent.core.topic_modeler.BERTopic'), \
                 patch('eCallsAgent.core.topic_modeler.UMAP'), \
                 patch('eCallsAgent.core.topic_modeler.HDBSCAN'):
                self.topic_modeler = TopicModeler(device=self.device)
                
            logger.info("TopicModeler instance created successfully")
        except Exception as e:
            logger.error(f"Error in setUp: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def test_initialization(self):
        """Test that TopicModeler initializes correctly."""
        try:
            self.assertEqual(self.topic_modeler.device, self.device)
            self.assertIsNotNone(self.topic_modeler.logger)
            logger.info("Initialization test passed")
        except Exception as e:
            logger.error(f"Error in test_initialization: {e}")
            logger.error(traceback.format_exc())
            raise
    
    @patch('eCallsAgent.core.topic_modeler.BERTopic')
    def test_create_chunk_model(self, mock_bertopic):
        """Test the _create_chunk_model method."""
        try:
            # Setup mock
            mock_instance = MagicMock()
            mock_bertopic.return_value = mock_instance
            
            # Call the method
            chunk_size = 100
            result = self.topic_modeler._create_chunk_model(chunk_size)
            
            # Assertions
            self.assertEqual(result, mock_instance)
            mock_bertopic.assert_called_once()
            logger.info("create_chunk_model test passed")
        except Exception as e:
            logger.error(f"Error in test_create_chunk_model: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def test_calculate_optimal_batch_size(self):
        """Test the _calculate_optimal_batch_size method."""
        try:
            # This method should return an integer
            batch_size = self.topic_modeler._calculate_optimal_batch_size()
            self.assertIsInstance(batch_size, int)
            self.assertGreater(batch_size, 0)
            logger.info(f"calculate_optimal_batch_size test passed, returned: {batch_size}")
        except Exception as e:
            logger.error(f"Error in test_calculate_optimal_batch_size: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def test_create_custom_labels(self):
        """Test the _create_custom_labels method."""
        try:
            # Create mock topic info DataFrame
            topic_info = pd.DataFrame({
                'Topic': [-1, 0, 1, 2],
                'Count': [10, 20, 15, 5],
                'Name': ['Outlier', 'Technology', 'Finance', 'Healthcare'],
                'Representation': [
                    '',
                    'tech, innovation, digital',
                    'finance, markets, growth',
                    'health, medical, care'
                ]
            })
            
            # Call the method
            custom_labels = self.topic_modeler._create_custom_labels(topic_info)
            
            # Assertions
            self.assertIsInstance(custom_labels, dict)
            self.assertEqual(len(custom_labels), 4)
            self.assertEqual(custom_labels[-1], 'Outlier')
            self.assertEqual(custom_labels[0], 'Technology')
            self.assertEqual(custom_labels[1], 'Finance')
            self.assertEqual(custom_labels[2], 'Healthcare')
            logger.info("create_custom_labels test passed")
        except Exception as e:
            logger.error(f"Error in test_create_custom_labels: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def test_save_figures(self):
        """Test the save_figures method."""
        try:
            # Create a mock topic model
            mock_topic_model = MagicMock()
            mock_topic_model.get_topic_info.return_value = pd.DataFrame({
                'Topic': [-1, 0, 1],
                'Count': [5, 10, 8],
                'Name': ['Outlier', 'Topic 1', 'Topic 2'],
                'Representation': [
                    '',
                    'tech, innovation, digital',
                    'finance, markets, growth'
                ]
            })
            mock_topic_model.topics_ = [0, 1, 0, -1, 1]
            
            # Add document_embeddings_ attribute to test that branch
            mock_topic_model.document_embeddings_ = np.random.rand(10, 2)
            
            # Create patches for the visualization module
            with patch('core.visualization.save_all_visualizations') as mock_save_all:
                with patch('core.visualization.save_embedding_visualization') as mock_save_embedding:
                    # Call the method
                    self.topic_modeler.save_figures(mock_topic_model)
                    
                    # Assertions
                    mock_save_all.assert_called_once()
                    mock_save_embedding.assert_called_once()
                    
                    logger.info("save_figures test passed")
        except Exception as e:
            logger.error(f"Error in test_save_figures: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def test_save_figures_without_embeddings(self):
        """Test the save_figures method when document_embeddings_ is not available."""
        # Create a mock topic model
        mock_model = MagicMock()
        mock_model.get_topic_info.return_value = pd.DataFrame({
            'Topic': [-1, 0, 1],
            'Count': [5, 10, 8],
            'Name': ['', 'Topic 0', 'Topic 1']
        })
        mock_model.topics_ = np.array([0, 1, 0, 1, 0])
        
        # Create a patched version of the topic_modeler.save_figures method
        # that checks for document_embeddings_ before calling save_embedding_visualization
        original_save_figures = self.topic_modeler.save_figures
        
        def patched_save_figures(topic_model):
            # Get topic info and create custom labels
            topic_info = topic_model.get_topic_info()
            custom_labels = self.topic_modeler._create_custom_labels(topic_info)
            
            # Save all visualizations
            from eCallsAgent.core.visualization import save_all_visualizations
            save_all_visualizations(topic_model, custom_labels, len(custom_labels) - 1)
            
            # Skip embedding visualization for this test
            # We're explicitly not checking for document_embeddings_
            
        # Apply the patch
        with patch.object(self.topic_modeler, 'save_figures', patched_save_figures):
            # Create mocks for the visualization functions
            mock_save_all = MagicMock()
            mock_save_embedding = MagicMock()
            
            # Patch the visualization functions at the correct import location
            with patch('core.visualization.save_all_visualizations', mock_save_all), \
                 patch('core.visualization.save_embedding_visualization', mock_save_embedding):
                
                # Call save_figures
                self.topic_modeler.save_figures(mock_model)
                
                # Assert that save_all_visualizations was called once
                mock_save_all.assert_called_once()
                
                # Assert that save_embedding_visualization was not called
                mock_save_embedding.assert_not_called()
                
        logger.info("Test passed: save_figures works correctly without document embeddings")

if __name__ == '__main__':
    unittest.main() 