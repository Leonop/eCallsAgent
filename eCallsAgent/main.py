"""
Main script for running the BERTopic analysis pipeline.
"""

import os
import sys
import logging
import traceback
import numpy as np
from eCallsAgent.config import global_options as gl
from eCallsAgent.core.data_handler import DataHandler
from eCallsAgent.core.embedding_generator import EmbeddingGenerator
from eCallsAgent.core.model_eval import ModelEvaluator
from eCallsAgent.core.topic_modeler import TopicModeler
from eCallsAgent.utils.cuda_setup import setup_cuda

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
        device = setup_cuda()
        logger.info(f"Using device: {device}")
        
        # Get absolute path to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(project_root, 'input_data', 'raw', gl.data_filename)
        data_handler = DataHandler(file_path, gl.YEAR_START, gl.YEAR_END)

        # Load or process documents
        docs_path = os.path.join(gl.input_folder, "processed", f'preprocessed_docs_{gl.YEAR_START}_{gl.YEAR_END}.txt')
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

        # Initialize model evaluator and topic modeler
        model_evaluator = ModelEvaluator()
        topic_modeler = TopicModeler(device)
        
        # Train baseline model
        baseline_model = topic_modeler.train_topic_model(docs, embeddings)
        
        # Evaluate baseline model
        baseline_coherence = model_evaluator.compute_coherence_score(baseline_model, docs)
        baseline_silhouette = model_evaluator.compute_silhouette_score(embeddings, baseline_model.topics_)
        
        logger.info("Baseline Model Evaluation:")
        logger.info(f"Coherence Score: {baseline_coherence:.4f}")
        logger.info(f"Silhouette Score: {baseline_silhouette:.4f}")
        logger.info(f"Number of Topics: {len(set(baseline_model.topics_)) - 1}")
        
        # Perform grid search and analyze results
        grid_search_results = model_evaluator.grid_search(docs, embeddings)
        model_evaluator.plot_parameter_effects(grid_search_results)
        
        # Generate and save topic information
        topic_info = topic_modeler.save_topic_keywords(baseline_model)
        topic_model = topic_modeler.update_topic_labels(topic_info, baseline_model)
        
        # Save the model
        os.makedirs(gl.model_folder, exist_ok=True)
        model_path = os.path.join(
            gl.model_folder, 
            f"bertopic_model_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}"
            f"_{topic_modeler.n_topics}_{gl.YEAR_START}_{gl.YEAR_END}.pkl"
        )
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