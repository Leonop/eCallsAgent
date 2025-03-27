"""
Test script to verify that patching works correctly.
"""

import os
import sys
import unittest
import logging
from unittest.mock import patch, MagicMock, call
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the mock topic modeler
from tests.mock_topic_modeler import save_figures
# Import TopicModeler class
from eCallsAgent.core.topic_modeler import TopicModeler

class TestPatching(unittest.TestCase):
    """Test cases for patching functions and methods."""
    
    def test_patching_imported_function(self):
        """Test patching an imported function."""
        # Create a mock function
        mock_func = MagicMock()
        
        # Patch the imported function
        with patch('tests.mock_topic_modeler.save_all_visualizations', mock_func):
            # Call the function that uses the imported function
            save_figures()
            
            # Assert that the mock function was called
            mock_func.assert_called_once()
            
            logger.info("✓ Patching imported function works correctly")
    
    def test_patching_module_function(self):
        """Test patching a function in a module."""
        # Create a mock topic model
        mock_model = MagicMock()
        mock_model.get_topic_info.return_value = pd.DataFrame({
            'Topic': [-1, 0, 1],
            'Count': [5, 10, 8],
            'Name': ['', 'Topic 0', 'Topic 1']
        })
        mock_model.topics_ = np.array([0, 1, 0, 1, 0])
        
        # Create a mock for the visualization function
        mock_func = MagicMock()
        
        # Patch the visualization module's save_all_visualizations function
        # The key fix is to patch at the correct import location
        with patch('core.visualization.save_all_visualizations', mock_func):
            # Create a topic modeler instance
            topic_modeler = TopicModeler(device="cpu")
            
            # Call save_figures
            topic_modeler.save_figures(mock_model)
            
            # Assert that the visualization function was called
            mock_func.assert_called_once()
            
        logger.info("✓ Patching module function works correctly")
    
    def test_patching_class_method(self):
        """Test patching a class method."""
        # Import the TopicModeler class
        from eCallsAgent.core.topic_modeler import TopicModeler
        
        # Create a mock method
        mock_method = MagicMock()
        
        # Patch the class method
        with patch.object(TopicModeler, '_create_custom_labels', mock_method):
            # Create an instance of the class
            topic_modeler = TopicModeler(device="cpu")
            
            # Create a mock topic model
            mock_topic_model = MagicMock()
            mock_topic_model.get_topic_info.return_value = MagicMock()
            
            # Call the method that uses the patched method
            with patch('core.visualization.save_all_visualizations'):
                with patch('core.visualization.save_embedding_visualization'):
                    topic_modeler.save_figures(mock_topic_model)
            
            # Assert that the mock method was called
            mock_method.assert_called_once()
            
            logger.info("✓ Patching class method works correctly")

if __name__ == '__main__':
    unittest.main() 