"""
Main script for running the BERTopic analysis pipeline.
"""

import os
# Set environment variables before importing any modules
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from numba import config
config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY = True
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
            else:
                logger.warning(f"Invalid embedding model index: {args.embedding_model}. Using default.")
        
        # Log the model being used
        model_key = gl.EMBEDDING_MODELS[gl.DEFAULT_MODEL_INDEX].replace('/', '-').replace(' ', '_')
        logger.info(f"Using embedding model: {model_key}")
        
        # Set up device
        _, device_str, _ = setup_cuda()
        logger.info(f"Using device: {device_str}")
        
        try:
            # Get absolute path to project root
            project_root = gl.PROJECT_DIR
            file_path = os.path.join(project_root, 'eCallsAgent', 'input_data', 'raw', gl.data_filename)
            
            # Load and preprocess data
            data_handler = DataHandler(file_path, gl.YEAR_START, gl.YEAR_END)
            processed_docs_path = os.path.join(gl.input_folder, "processed", f'componenttext_{gl.YEAR_START}_{gl.YEAR_END}_{model_key}.txt')

            if os.path.exists(processed_docs_path):
                logger.info(f"Found preprocessed docs at {processed_docs_path}. Loading...")
                docs = data_handler.load_doc_parallel(processed_docs_path)
            else:
                logger.info("Processed docs not found. Processing raw data...")
                data_df = data_handler.load_data() # load the raw data
                logger.info(f"process duplicate earnings calls {data_df.shape}")
                docs = data_handler.process_dup_earnings_calls(data_df, processed_docs_path) # remove duplicate earnings calls
                os.makedirs(gl.output_folder, exist_ok=True)
                logger.info(f"Processed and saved docs to {processed_docs_path}")
            
            # Initialize embedding generator
            logger.info(f"embedding generator initialized")
            embedding_gen = EmbeddingGenerator(device_str, gl.DEFAULT_MODEL_INDEX)
            
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
            
            if not gl.SKIP_GRID_SEARCH:
                logger.info("Starting grid search to find optimal parameters...")
                model_evaluator = ModelEvaluator()
                best_model, best_params = model_evaluator.grid_search(docs, embeddings)
                
                if best_model is not None:
                    logger.info(f"Grid search completed. Best parameters found: {best_params}")
                    final_model = best_model
                else:
                    logger.warning("Grid search failed or found no valid parameters. Using default parameters.")
                    # Proceed with default topic modeling as before
                    topic_modeler = TopicModeler(device_str)
                    final_model = topic_modeler.train_topic_model(docs, embeddings)
            else:
                logger.info("Skipping grid search, using default parameters")
                topic_modeler = TopicModeler(device_str)
                final_model = topic_modeler.train_topic_model(docs, embeddings)
                                    
            # Map all documents to topics
            logger.info("************** Create topic probabilities CSV **************")
            output_path = os.path.join(gl.output_folder, f"topic_probabilities_{gl.YEAR_START}_{gl.YEAR_END}_{model_key}.csv")
            
            try:
                # Check if data variable exists and is accessible
                if 'data_df' in locals() and data_df is not None:
                    logger.info(f" The size of the data variable is {data_df.shape}")
                    data_handler._create_topic_probabilities_csv(data_df, docs, embeddings, topic_modeler, output_path)
                else:
                    logger.warning("Data variable not available. This may happen during small sample tests or when data loading is bypassed.")
                    logger.warning("Skipping topic probabilities CSV creation.")
            except Exception as e:
                logger.error(f"Error creating topic probabilities CSV: {e}")
                logger.error(traceback.format_exc())
                logger.warning("Continuing with rest of processing despite CSV creation error.")

            # Evaluate model
            model_evaluator = ModelEvaluator()
            baseline_coherence = model_evaluator.compute_coherence_score(final_model, docs)
            baseline_silhouette = model_evaluator.compute_silhouette_score(embeddings, final_model.topics_)
            
            logger.info("Model Evaluation:")
            logger.info(f"Coherence Score: {baseline_coherence:.4f}")
            logger.info(f"Silhouette Score: {baseline_silhouette:.4f}")
            logger.info(f"Number of Topics: {len(set(final_model.topics_)) - 1}")
            logger.info(f"Mean Topic Confidence Score: {topic_modeler.confidence_topic_score()[0]:.4f}")
            logger.info(f"Standard Deviation of Topic Confidence Score: {topic_modeler.confidence_topic_score()[1]:.4f}")
            
            # parameters for the model
            # final_model_params = topic_modeler._calculate_adaptive_parameters(docs, embeddings)
            n_neighbors = gl.final_parameters['n_neighbors']
            n_components = gl.final_parameters['n_components']
            min_cluster_size = gl.final_parameters['min_cluster_size']
            min_samples = gl.final_parameters['min_samples']
            n_topics = topic_modeler.n_topics

            # Save the model
            os.makedirs(gl.models_folder, exist_ok=True)
            model_path = os.path.join(
                gl.models_folder, 
                f"bertopic_model_{n_neighbors}_{n_components}_{min_cluster_size}_{min_samples}_{n_topics}_{gl.YEAR_START}_{gl.YEAR_END}_{model_key}.pkl"
            )
            final_model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save visualization figures
            topic_modeler.save_figures(final_model, include_doc_vis=True)
            topic_info = final_model.get_topic_info()
            logger.info(f"Generated {len(topic_info)} topics")
            
            # Save UMAP results if available
            if hasattr(final_model, 'umap_model') and final_model.umap_model is not None:
                # Save UMAP results
                umap_file = os.path.join(gl.output_folder, 'umap_embeddings.npy')
                if not os.path.exists(umap_file):
                    logger.info("Saving UMAP embeddings for future use")
                    umap_embeddings = final_model.umap_model.transform(embeddings)
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