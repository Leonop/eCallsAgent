#!/usr/bin/env python
# -*- coding: utf-8 -*-
# written by: zicheng.leo@gmail.com
# modified by: assistant for debugging
# description: Test script for eCallsAgent package

"""
Debug test script for eCallsAgent package to verify core functionality.
"""

import os
import sys
import logging
import traceback
import numpy as np
import json
import re
from unittest.mock import patch, MagicMock

from eCallsAgent.utils.cuda_setup import setup_cuda
from eCallsAgent.utils.logging_utils import setup_logging
from eCallsAgent.core.data_handler import DataHandler
from eCallsAgent.core.embedding_generator import EmbeddingGenerator
from eCallsAgent.core.topic_modeler import TopicModeler
from eCallsAgent.core.visualization import save_figure
import eCallsAgent.config.global_options as gl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('test_eCallsAgent.log'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def create_mock_data():
    """Create small mock data for testing."""
    # Create 20 simple documents for testing
    mock_docs = [
        "The company reported increased revenue for this quarter",
        "Our financial performance was strong with growth in all segments",
        "New product launches drove significant sales increases",
        "We experienced challenges in the supply chain this quarter",
        "Market conditions remain favorable for our business",
        "Research and development investments continue to yield results",
        "Customer feedback has been positive on our latest offerings",
        "We expanded our operations into new global markets",
        "Cost reduction initiatives have improved our margins",
        "The board approved a new dividend policy for shareholders",
        "Regulatory changes affected our business in certain regions",
        "We hired key talent to strengthen our leadership team",
        "Strategic partnerships have opened new revenue streams",
        "Technology upgrades improved our operational efficiency",
        "We acquired a company to complement our product line",
        "Sustainability initiatives reduced our environmental impact",
        "Customer acquisition costs decreased in the last quarter",
        "We launched a new digital platform for our customers",
        "Marketing campaigns resulted in higher brand awareness",
        "Supply chain improvements reduced our delivery times"
    ]
    
    # Create mock embeddings (dimension 384 matches many common models)
    mock_embeddings = np.random.rand(len(mock_docs), 384)
    
    return mock_docs, mock_embeddings

def create_required_directories():
    """Create all the necessary directories for testing."""
    # Main directories
    directories = [
        gl.output_folder,
        gl.figures_folder,
        gl.models_folder,
        gl.temp_folder,
        gl.log_folder,
        gl.embeddings_folder,
        os.path.join(gl.temp_folder, "topic_models"),
        os.path.join(gl.output_folder, "temp"),
        os.path.join(gl.output_folder, "visualizations")
    ]
    
    for directory in directories:
        logger.info(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)

def create_mock_cache():
    """Create a mock cache file for topic labels."""
    cache_file = os.path.join(gl.output_folder, 'temp', 'topic_label_cache.json')
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    # Create a simple cache with some example labels
    mock_cache = {
        "revenue_performance_financial": ["Financial Performance", "Revenue and Growth"],
        "market_conditions_business": ["Market Analysis", "Business Environment"],
        "supply_chain_operations_logistics": ["Operations", "Supply Chain Management"]
    }
    
    logger.info(f"Creating mock cache file at {cache_file}")
    with open(cache_file, 'w') as f:
        json.dump(mock_cache, f)

# Mock for OpenAI client
class MockOpenAIClient:
    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = self.mock_create

    def mock_create(self, model, messages, temperature, max_tokens):
        # Create a mock response based on the prompt
        # Extract keywords from the prompt
        prompt = messages[1]["content"]
        keywords_match = re.search(r'Primary Keywords\*\*:(.*?)(?:-|\n)', prompt, re.DOTALL)
        keywords = []
        if keywords_match:
            keywords_text = keywords_match.group(1).strip()
            keywords = [k.strip() for k in keywords_text.split(',') if k.strip()]
        
        # Create a mock response
        if keywords:
            topic = keywords[0].title()
            subtopic = ' '.join(keywords[1:3]).title() if len(keywords) > 1 else "General"
            mock_response = f"Topic: {topic}, Subtopic: {subtopic}"
        else:
            mock_response = "Topic: Business Topic, Subtopic: General Business"
        
        # Create mock response object
        response = MagicMock()
        message = MagicMock()
        message.content = mock_response
        choice = MagicMock()
        choice.message = message
        response.choices = [choice]
        
        return response

def main() -> None:
    """Test main processing pipeline with debugging."""
    try:
        # Create required directories
        create_required_directories()
        
        # Create mock cache for topic labels
        create_mock_cache()
        
        # Initialize logging
        logger = setup_logging(gl.log_folder)
        logger.info("Starting eCallsAgent debug test")
        
        # Setup device
        device = setup_cuda()
        logger.info(f"Using device: {device}")
        
        # Create mock data
        logger.info("Creating mock data for testing")
        docs, embeddings = create_mock_data()
        logger.info(f"Created {len(docs)} mock documents with embeddings shape {embeddings.shape}")
        
        # Test TopicModeler with small data
        logger.info("Testing TopicModeler with mock data")
        
        # Set smaller parameters for quick testing
        gl.N_NEIGHBORS = [5]
        gl.N_COMPONENTS = [5]
        gl.MIN_CLUSTER_SIZE = [2]
        
        # Mock OpenAI integration
        with patch('eCallsAgent.core.topic_modeler.OpenAI', return_value=MockOpenAIClient()):
            topic_modeler = TopicModeler(device)
            try:
                # Explicitly mock the OpenAI setup method
                topic_modeler._setup_openai = MagicMock()
                topic_modeler.openai_available = True
                topic_modeler.client = MockOpenAIClient()
                
                # Train a basic model with the mock data
                logger.info("Training topic model with mock data")
                try:
                    topic_model = topic_modeler.train_topic_model(
                        docs, 
                        embeddings,
                        save_visualizations=False  # Disable visualizations initially for faster testing
                    )
                    logger.info("Topic model training completed")
                    
                    # Test saving topic keywords
                    logger.info("Testing save_topic_keywords")
                    topic_info = topic_modeler.save_topic_keywords(topic_model)
                    logger.info(f"Generated topic info with {len(topic_info)} rows")
                    
                    # Print some debug info about the topic info
                    logger.info(f"Topic info columns: {list(topic_info.columns)}")
                    if not topic_info.empty:
                        logger.info(f"First topic: {topic_info.iloc[0].to_dict()}")
                    
                    # Test updating topic labels with more robustness
                    logger.info("Testing update_topic_labels")
                    try:
                        updated_model = topic_modeler.update_topic_labels(topic_info, topic_model)
                        logger.info("Topic labels updated successfully")
                    except Exception as label_err:
                        logger.error(f"Error updating topic labels: {label_err}")
                        logger.error(traceback.format_exc())
                        # Continue with the original model
                        updated_model = topic_model
                        
                    # Save model
                    model_path = os.path.join(gl.models_folder, "test_model.pkl")
                    updated_model.save(model_path)
                    logger.info(f"Saved test model to {model_path}")
                    
                    # Test basic functionality passed
                    logger.info("✅ Basic topic modeling test passed")
                    
                    # Test visualization module
                    try:
                        import plotly.graph_objects as go
                        logger.info("Testing visualization module")
                        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
                        save_figure(fig, "test_figure", n_topics=len(set(updated_model.topics_)))
                        logger.info("✅ Visualization test passed")
                    except Exception as viz_err:
                        logger.error(f"❌ Visualization test failed: {viz_err}")
                        logger.error(traceback.format_exc())
                    
                    # Try saving a visualization from the model
                    try:
                        logger.info("Testing model visualization")
                        topic_modeler.save_figures(updated_model, specific_visualizations=["barchart"])
                        logger.info("✅ Model visualization test passed")
                    except Exception as model_viz_err:
                        logger.error(f"❌ Model visualization test failed: {model_viz_err}")
                        logger.error(traceback.format_exc())
                    
                    logger.info("✅ eCallsAgent package test completed successfully")
                    
                except Exception as model_err:
                    logger.error(f"❌ Topic modeling test failed: {model_err}")
                    logger.error(traceback.format_exc())
                    # Print stack trace
                    import traceback
                    traceback.print_exc()
                    
            except Exception as topic_err:
                logger.error(f"❌ TopicModeler initialization failed: {topic_err}")
                logger.error(traceback.format_exc())
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 