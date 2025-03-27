"""
Tests for topic modeling functionality.
"""

import unittest
import os
import torch
import numpy as np
from eCallsAgent.core.topic_modeler import TopicModeler
import logging
from umap import UMAP
from hdbscan import HDBSCAN
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestTopicModeling(unittest.TestCase):
    def setUp(self):
        """Initialize test environment."""
        try:
            # Set device
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                
            logger.info(f"Test running on device: {self.device}")
            
            # Initialize test data
            self.test_docs = [
                "This is document one about finance.",
                "Document two discusses earnings.",
                "Third document about market trends.",
                "Fourth document about technology.",
                "Fifth document about investments.",
                "Sixth document about market analysis.",
                "Seventh document about financial planning.",
                "Eighth document about stock trading.",
                "Ninth document about economic trends.",
                "Tenth document about business strategy."
            ] * 3  # 30 documents total
            
            # Create test embeddings
            self.embedding_dim = 768
            self.test_embeddings = np.random.rand(len(self.test_docs), self.embedding_dim).astype(np.float32)
            
            # Print environment info
            logger.info("\nTest environment setup:")
            logger.info(f"CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                
        except Exception as e:
            logger.error(f"Error in setUp: {str(e)}")
            raise
    
    def test_topic_modeler_initialization(self):
        """Test if topic modeler initializes correctly."""
        topic_modeler = TopicModeler(self.device)
        self.assertIsNotNone(topic_modeler)
        self.assertEqual(topic_modeler.device, self.device)
    
    def test_model_training(self):
        """Test if model trains correctly."""
        try:
            # Initialize topic modeler
            topic_modeler = TopicModeler(self.device)
            
            # Configure model parameters for testing
            topic_modeler.topic_model.min_topic_size = 3
            topic_modeler.topic_model.calculate_probabilities = False
            
            # Update UMAP parameters
            topic_modeler.umap_model = UMAP(
                n_neighbors=5,
                n_components=2,
                min_dist=0.1,
                metric='cosine',
                random_state=42,
                verbose=False,
                n_jobs=1,
                low_memory=True,
                output_metric='euclidean'
            )
            
            # Update HDBSCAN parameters
            topic_modeler.hdbscan_model = HDBSCAN(
                min_samples=2,
                min_cluster_size=3,
                metric='euclidean',
                prediction_data=True,
                core_dist_n_jobs=1,
                algorithm='generic'
            )
            
            # Train model
            try:
                logger.info(f"Training model on device: {self.device}")
                logger.info(f"Test docs length: {len(self.test_docs)}")
                logger.info(f"Test embeddings shape: {self.test_embeddings.shape}")
                model = topic_modeler.train_topic_model(self.test_docs, self.test_embeddings)
            except Exception as e:
                logger.error(f"Error during model training: {str(e)}")
                logger.error(f"Test docs length: {len(self.test_docs)}")
                logger.error(f"Test embeddings shape: {self.test_embeddings.shape}")
                raise
            
            # Verify model outputs
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, 'topics_'))
            
            # Check if number of topics is reasonable
            n_topics = len(set(model.topics_)) - 1  # Exclude -1 (noise)
            self.assertGreater(n_topics, 0)
            
        except Exception as e:
            logger.error(f"Error in test_model_training: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
    
    def test_end_to_end_processing(self):
        """Test the entire topic modeling pipeline."""
        try:
            # Initialize topic modeler with appropriate parameters
            topic_modeler = TopicModeler(self.device)
            
            # Configure topic model parameters
            topic_modeler.topic_model.min_topic_size = 3
            topic_modeler.topic_model.calculate_probabilities = False
            
            # Update UMAP parameters
            topic_modeler.umap_model = UMAP(
                n_neighbors=5,
                n_components=2,
                min_dist=0.1,
                metric='cosine',
                random_state=42,
                verbose=False,
                n_jobs=1,
                low_memory=True,
                output_metric='euclidean',
                transform_queue_size=1
            )
            
            # Update HDBSCAN parameters
            topic_modeler.hdbscan_model = HDBSCAN(
                min_samples=2,
                min_cluster_size=3,
                metric='euclidean',
                prediction_data=True,
                core_dist_n_jobs=1,
                algorithm='generic'
            )
            
            # Train model
            try:
                logger.info(f"Training model on device: {self.device}")
                logger.info(f"Test docs length: {len(self.test_docs)}")
                logger.info(f"Test embeddings shape: {self.test_embeddings.shape}")
                model = topic_modeler.train_topic_model(self.test_docs, self.test_embeddings)
            except Exception as e:
                logger.error(f"Error during model training: {str(e)}")
                logger.error(f"Test docs length: {len(self.test_docs)}")
                logger.error(f"Test embeddings shape: {self.test_embeddings.shape}")
                raise
            
            # Verify model outputs
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, 'topics_'))
            
            # Check if topics were generated
            n_topics = len(set(model.topics_)) - 1  # Exclude -1 (noise)
            self.assertGreater(n_topics, 0)
            logger.info(f"Generated {n_topics} topics")
            
        except Exception as e:
            logger.error(f"Error in test_end_to_end_processing: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
    
    def tearDown(self):
        """Clean up after each test."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error in tearDown: {str(e)}")

if __name__ == '__main__':
    unittest.main() 