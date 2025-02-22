#!/usr/bin/env python
# -*- coding: utf-8 -*-
# written by: zicheng.leo@gmail.com
# date: 2025-02-12
# version: 0.1
# description: Main script for BERTopic processing pipeline to generate business fundamentals topic model

"""
Main script for BERTopic processing pipeline.
"""

import os
import sys
import traceback
import logging

from eCallsAgent.utils.cuda_setup import setup_cuda
from eCallsAgent.utils.logging_utils import setup_logging, log_system_info, log_gpu_info
from eCallsAgent.core.data_handler import DataHandler
from eCallsAgent.core.embedding_generator import EmbeddingGenerator
from eCallsAgent.core.topic_modeler import TopicModeler
import eCallsAgent.config.global_options as gl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('bertopic_processing.log'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main processing pipeline."""
    try:
        # Initialize logging
        logger = setup_logging(gl.log_folder)
        log_system_info()
        
        device = setup_cuda()
        logger.info(f"Using device: {device}")
        
        # Get absolute path to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(project_root, 'eCallsAgent', 'input_data', 'raw', gl.data_filename)
        data_handler = DataHandler(file_path, gl.YEAR_START, gl.YEAR_END)

        # Load or process documents
        docs_path = os.path.join(gl.output_folder, f'preprocessed_docs_{gl.YEAR_START}_{gl.YEAR_END}.txt')
        if os.path.exists(docs_path):
            logger.info(f"Found preprocessed docs at {docs_path}. Loading...")
            docs = data_handler.load_doc_parallel(docs_path)
        else:
            logger.info("Processed docs not found. Processing raw data...")
            data = data_handler.load_data()
            docs = data_handler.preprocess_text(data)
            os.makedirs(gl.output_folder, exist_ok=True)
            logger.info(f"Saving processed docs to {docs_path}")
            with open(docs_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(docs))

        # Initialize embedding generator
        embedding_gen = EmbeddingGenerator(device)
        
        # Try to load existing embeddings first
        try:
            logger.info("Attempting to load existing embeddings...")
            embeddings = embedding_gen.load_embeddings(gl.YEAR_START, gl.YEAR_END)
            logger.info(f"Successfully loaded embeddings with shape {embeddings.shape}")
        except FileNotFoundError:
            logger.info("No existing embeddings found. Generating new embeddings...")
            embeddings = embedding_gen.generate_embeddings(docs)

        topic_modeler = TopicModeler(device)
        topic_model = topic_modeler.train_topic_model(docs, embeddings)

        # First save topic info and generate labels
        topic_info = topic_modeler.save_topic_keywords(topic_model)
        
        # Update model with topic and subtopic labels
        topic_model = topic_modeler.update_topic_labels(topic_info, topic_model)
        
        # Now save the model with the generated labels
        os.makedirs(gl.model_folder, exist_ok=True)
        model_path = os.path.join(gl.model_folder, 
            f"bertopic_model_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{topic_modeler.n_topics}_{gl.YEAR_START}_{gl.YEAR_END}.pkl")
        topic_model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save visualization figures
        topic_modeler.save_figures(topic_model)

        logger.info(f"Generated {len(topic_info)} topics")
        logger.info("Topic modeling completed successfully")
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 