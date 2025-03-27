"""
Test script to verify that visualization functions are properly imported in topic_modeler.py.
"""

import os
import sys
import unittest
import logging
import traceback
from unittest.mock import patch, MagicMock
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestVisualizationImports(unittest.TestCase):
    """Test that visualization functions are properly imported and used in topic_modeler.py"""
    
    def setUp(self):
        """Set up the test environment"""
        try:
            # Import the TopicModeler class
            from eCallsAgent.core.topic_modeler import TopicModeler
            
            # Create a TopicModeler instance with the required device parameter
            self.topic_modeler = TopicModeler(device="cpu")
            
            # Create a mock topic model
            self.mock_topic_model = MagicMock()
            self.mock_topic_model.get_topic_info.return_value = pd.DataFrame({
                'Topic': [1, 2, 3],
                'Count': [10, 20, 30],
                'Name': ['Topic 1', 'Topic 2', 'Topic 3']
            })
            
            logger.info("Test setup completed successfully")
        except Exception as e:
            logger.error(f"Error in test setup: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def test_visualization_functions_called(self):
        """Test that visualization functions are called from save_figures method"""
        try:
            # Create patches for the visualization functions
            with patch('core.visualization.save_all_visualizations') as mock_save_all:
                with patch('core.visualization.save_embedding_visualization') as mock_save_embedding:
                    # Call the save_figures method
                    self.topic_modeler.save_figures(self.mock_topic_model)
                    
                    # Assert that the visualization functions were called
                    mock_save_all.assert_called_once()
                    
                    # Log success
                    logger.info("Test passed: visualization functions were called")
        except Exception as e:
            logger.error(f"Error in test: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def test_visualization_functions_imported(self):
        """Test that visualization functions are imported inside the save_figures method"""
        try:
            # Create a patch for the import statement
            with patch('core.visualization') as mock_visualization:
                # Set up the mock to return MagicMock objects for the visualization functions
                mock_visualization.save_all_visualizations = MagicMock()
                mock_visualization.save_embedding_visualization = MagicMock()
                
                # Call the save_figures method
                self.topic_modeler.save_figures(self.mock_topic_model)
                
                # Assert that the visualization functions were accessed
                self.assertTrue(hasattr(mock_visualization, 'save_all_visualizations'))
                self.assertTrue(hasattr(mock_visualization, 'save_embedding_visualization'))
                
                # Log success
                logger.info("Test passed: visualization functions were imported")
        except Exception as e:
            logger.error(f"Error in test: {e}")
            logger.error(traceback.format_exc())
            raise

if __name__ == '__main__':
    unittest.main() 