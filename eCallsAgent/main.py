"""
Main script for running the BERTopic analysis pipeline.
"""

import os
# Set tokenizer parallelism before importing any HuggingFace modules
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import logging
import traceback
import argparse
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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the topic modeling pipeline')
    
    # Add arguments
    parser.add_argument('--skip_grid_search', action='store_true', 
                        help='Skip grid search and use default parameters')
    parser.add_argument('--parameter_set', type=str, default='default',
                        choices=['default', 'more_topics', 'fewer_topics', 'best'],
                        help='Parameter set to use when skipping grid search')
    parser.add_argument('--year_start', type=int, default=None,
                        help='Start year for data filtering')
    parser.add_argument('--year_end', type=int, default=None,
                        help='End year for data filtering')
    parser.add_argument('--embedding_model', type=int, default=1,
                        help='Index of embedding model to use from EMBEDDING_MODELS list')
    
    args = parser.parse_args()
    
    # Log the arguments
    logger.info(f"Command line arguments: {args}")
    
    return args

def main() -> None:
    """Main processing pipeline with distributed computing support."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Override global settings if command line arguments are provided
        if args.skip_grid_search:
            gl.SKIP_GRID_SEARCH = True
            logger.info(f"Grid search will be skipped as specified by command line argument.")
        
        if args.parameter_set:
            gl.PARAMETER_SET = args.parameter_set
            logger.info(f"Using parameter set: {gl.PARAMETER_SET}")
        
        if args.year_start:
            gl.YEAR_START = args.year_start
            logger.info(f"Using start year: {gl.YEAR_START}")
        
        if args.year_end:
            gl.YEAR_END = args.year_end
            logger.info(f"Using end year: {gl.YEAR_END}")
            
        if args.embedding_model is not None:
            # Ensure the index is valid
            if 0 <= args.embedding_model < len(gl.EMBEDDING_MODELS):
                gl.DEFAULT_MODEL_INDEX = args.embedding_model
                logger.info(f"Using embedding model: {gl.EMBEDDING_MODELS[gl.DEFAULT_MODEL_INDEX]}")
            else:
                logger.warning(f"Invalid embedding model index: {args.embedding_model}. Using default.")
        
        # Set up device
        cuda_ready, device_str, memory_gb = setup_cuda()
        logger.info(f"Using device: {device_str}")
        
        try:
            # Get absolute path to project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(project_root, 'eCallsAgent', 'input_data', 'raw', gl.data_filename)
            
            # Load and preprocess data
            data_handler = DataHandler(file_path, gl.YEAR_START, gl.YEAR_END)
            docs_path = os.path.join(gl.input_folder, "processed", f'preprocessed_docs_{gl.YEAR_START}_{gl.YEAR_END}.txt')
            
            # Create processed directory if it doesn't exist
            os.makedirs(os.path.dirname(docs_path), exist_ok=True)
            
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
            embedding_gen = EmbeddingGenerator(device_str)
            
            # Try to load existing embeddings first
            try:
                logger.info("Attempting to load existing embeddings...")
                embeddings = embedding_gen.load_embeddings(gl.YEAR_START, gl.YEAR_END)
                logger.info(f"Successfully loaded embeddings with shape {embeddings.shape}")
                
                # Check if embeddings match number of documents
                if embeddings.shape[0] != len(docs):
                    logger.warning(f"Mismatch between number of documents ({len(docs)}) and embeddings ({embeddings.shape[0]})")
                    logger.info("Regenerating embeddings to match documents...")
                    embeddings = embedding_gen.generate_embeddings(docs)
            except FileNotFoundError:
                logger.info("No existing embeddings found. Generating new embeddings...")
                embeddings = embedding_gen.generate_embeddings(docs)
            
            # Initialize and train topic model
            topic_modeler = TopicModeler(device_str)
            model = topic_modeler.train_topic_model(docs, embeddings)
            
            # Evaluate model
            model_evaluator = ModelEvaluator()
            baseline_coherence = model_evaluator.compute_coherence_score(model, docs)
            baseline_silhouette = model_evaluator.compute_silhouette_score(embeddings, model.topics_)
            
            logger.info("Model Evaluation:")
            logger.info(f"Coherence Score: {baseline_coherence:.4f}")
            logger.info(f"Silhouette Score: {baseline_silhouette:.4f}")
            logger.info(f"Number of Topics: {len(set(model.topics_)) - 1}")
            
            # Save results
            topic_info = topic_modeler.save_topic_keywords(model)
            topic_model = topic_modeler.update_topic_labels(topic_info, model)
            
            # Save the model
            os.makedirs(gl.models_folder, exist_ok=True)
            model_path = os.path.join(
                gl.models_folder, 
                f"bertopic_model_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}"
                f"_{topic_modeler.n_topics}_{gl.YEAR_START}_{gl.YEAR_END}.pkl"
            )
            topic_model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save visualization figures
            topic_modeler.save_figures(topic_model)
            logger.info(f"Generated {len(topic_info)} topics")
            
            # Save UMAP results if available
            if hasattr(topic_model, 'umap_model') and topic_model.umap_model is not None:
                # Save UMAP results
                umap_file = os.path.join(gl.output_folder, 'umap_embeddings.npy')
                if not os.path.exists(umap_file):
                    logger.info("Saving UMAP embeddings for future use")
                    umap_embeddings = topic_model.umap_model.transform(embeddings)
                    np.save(umap_file, umap_embeddings)
            
            logger.info("Topic modeling completed successfully")
            
        except Exception as e:
            logger.error(f"Error in main process: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()