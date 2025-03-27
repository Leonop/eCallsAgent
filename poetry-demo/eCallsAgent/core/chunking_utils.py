"""
Utility functions for topic modeling chunk processing.
This module contains functions for parallel processing of document chunks.
"""

import os
# Set Numba CUDA compatibility environment variable
import numpy as np
import logging
import traceback
import torch
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from eCallsAgent.config import global_options as gl
from eCallsAgent.config.default_model_params import BEST_UMAP_PARAMS, BEST_HDBSCAN_PARAMS
from typing import List, Union

# Try to import GPU-accelerated versions if available
CUML_AVAILABLE = False
CUDA_READY = False

try:
    # Check CUDA first
    CUDA_READY = torch.cuda.is_available()
    
    # Only try to import cuML if CUDA is ready
    if CUDA_READY:
        try:
            from cuml import UMAP as cumlUMAP
            from cuml.cluster import HDBSCAN as cumlHDBSCAN
            CUML_AVAILABLE = True
            logging.getLogger(__name__).info("Successfully imported GPU-accelerated models (cuML)")
        except ImportError:
            CUML_AVAILABLE = False
            logging.getLogger(__name__).warning("GPU-accelerated models (cuML) not available despite CUDA being ready")
        except Exception as e:
            CUML_AVAILABLE = False
            logging.getLogger(__name__).warning(f"Error importing GPU-accelerated models: {e}")
except Exception as e:
    CUDA_READY = False
    logging.getLogger(__name__).warning(f"Error checking CUDA availability: {e}")
    logging.getLogger(__name__).warning(traceback.format_exc())


# add logger
logger = logging.getLogger(__name__)

def process_chunk_worker(chunk_idx: int, chunk_docs: List[str], chunk_embeddings: np.ndarray,
                        total_chunks: int,
                        _cpu: bool = True
                        ):
    """Process a chunk of documents for topic modeling
    For Chunking, we use CPU-based models.
    
    Args:
        chunk_idx: Index of the current chunk
        chunk_docs: List of documents in the chunk
        chunk_embeddings: Embeddings for the documents
        total_chunks: Total number of chunks being processed
        
    Returns:
        Dictionary mapping topic keys to representative documents and embeddings
    """
    try:
        # Setup logging
        logger = logging.getLogger(__name__)
        logger.info(f"Worker processing chunk {chunk_idx+1}/{total_chunks} with {len(chunk_docs)} documents")
        
        # Use best parameters from default_model_params
        n_neighbors = BEST_UMAP_PARAMS['n_neighbors']
        n_components = BEST_UMAP_PARAMS['n_components']
        min_dist = BEST_UMAP_PARAMS['min_dist']
        min_samples = BEST_HDBSCAN_PARAMS['min_samples']
        min_cluster_size = BEST_HDBSCAN_PARAMS['min_cluster_size']
        
        # Log the parameters being used
        logger.info(f"Using UMAP parameters: n_neighbors={n_neighbors}, n_components={n_components}, min_dist={min_dist}")
        logger.info(f"Using HDBSCAN parameters: min_samples={min_samples}, min_cluster_size={min_cluster_size}")
        
        # Use CPU based topic model to generate topic model
        umap_model, hdbscan_model = _cpu_topic_model(n_neighbors, n_components, min_dist, min_samples, min_cluster_size)
        
        # Create a BERTopic model instance for this worker
        chunk_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            embedding_model=None,
            calculate_probabilities=False,
            verbose=False
        )
        
        # Convert any GPU arrays to CPU arrays before parallel processing
        if hasattr(chunk_embeddings, 'get'):
            chunk_embeddings = chunk_embeddings.get()
        
        # Fit models
        transformed_embeddings = umap_model.fit_transform(chunk_embeddings)
        hdbscan_model.fit(transformed_embeddings)
        topics = hdbscan_model.labels_
        
        # Update the model's internal state
        chunk_model.topics_ = topics
        
        # Get number of topics
        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        logger.info(f"Worker found {n_topics} topics in chunk {chunk_idx+1}")
        
        # If very few topics, retry with more aggressive parameters
        if n_topics < 5:
            logger.warning(f"Worker: Few topics found ({n_topics}), retrying with more aggressive parameters")
            
            retry_umap = UMAP(
                n_neighbors=5,
                n_components=75,
                min_dist=0.0,
                metric='cosine',
                random_state=42,
                verbose=False,
                low_memory=True,
                n_jobs=1
            )
            
            retry_hdbscan = HDBSCAN(
                min_samples=3,
                min_cluster_size=15,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True,
                core_dist_n_jobs=1
            )
            
            retry_transformed = retry_umap.fit_transform(chunk_embeddings)
            retry_hdbscan.fit(retry_transformed)
            retry_topics = retry_hdbscan.labels_
            retry_n_topics = len(set(retry_topics)) - (1 if -1 in retry_topics else 0)
            
            if retry_n_topics > n_topics:
                topics = retry_topics
                transformed_embeddings = retry_transformed
                n_topics = retry_n_topics
                chunk_model.topics_ = topics
                logger.info(f"Worker: Retry found {n_topics} topics in chunk {chunk_idx+1}")
        
        # Create a topic-to-document mapping
        topic_to_docs = {}
        for doc_idx, topic_label in enumerate(topics):
            if topic_label not in topic_to_docs:
                topic_to_docs[topic_label] = []
            topic_to_docs[topic_label].append(doc_idx)
        
        # Collect representatives for each topic
        chunk_representatives = {}
        
        for topic_id, doc_indices in topic_to_docs.items():
            if topic_id == -1:  # Skip outlier topic
                continue
            
            # Take up to MAX_DOCS_PER_TOPIC documents
            max_docs = min(gl.MAX_DOCS_PER_TOPIC, len(doc_indices))
            selected_indices = doc_indices[:max_docs]
            
            # Collect documents and embeddings
            topic_docs = [chunk_docs[idx] for idx in selected_indices]
            topic_embeddings = [chunk_embeddings[idx].tolist() for idx in selected_indices]
            
            # Create topic key
            topic_key = f"topic_{chunk_idx}_{topic_id}"
            
            # Store representatives
            chunk_representatives[topic_key] = {
                'docs': topic_docs,
                'embeddings': topic_embeddings
            }
        
        # Convert any GPU arrays to CPU arrays before updating
        if chunk_representatives:
            for topic_key, topic_data in chunk_representatives.items():
                if 'embeddings' in topic_data:
                    # Convert GPU arrays to CPU arrays
                    topic_data['embeddings'] = [emb.get() if hasattr(emb, 'get') else emb for emb in topic_data['embeddings']]
        
        # Clear memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Return the representatives for this chunk
        return chunk_representatives
        
    except Exception as e:
        logger.error(f"Worker error processing chunk {chunk_idx+1}: {e}")
        logger.error(traceback.format_exc())
        return {} 
    

def _cpu_topic_model(n_neighbors, n_components, min_dist, min_samples, min_cluster_size, cluster_selection_epsilon=0.2):
    """Create CPU-based UMAP and HDBSCAN models.
    
    Args:
        n_neighbors: Number of neighbors for UMAP
        n_components: Number of dimensions for UMAP
        min_dist: Minimum distance for UMAP
        min_samples: Minimum samples for HDBSCAN
        min_cluster_size: Minimum cluster size for HDBSCAN
        cluster_selection_epsilon: Epsilon for cluster selection in HDBSCAN
        
    Returns:
        tuple: (umap_model, hdbscan_model)
    """
    logger.info("Using CPU-based UMAP and HDBSCAN for chunk processing")
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric='cosine',
        random_state=42,
        verbose=False,
        low_memory=True,
        n_jobs=-1
    )
    
    hdbscan_model = HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
        core_dist_n_jobs=-1
    )
    return umap_model, hdbscan_model

def _gpu_topic_model(n_neighbors, n_components, min_dist, min_samples, min_cluster_size, cluster_selection_epsilon=0.2):
    """Create GPU-accelerated UMAP and HDBSCAN models using cuML.
    
    Args:
        n_neighbors: Number of neighbors for UMAP
        n_components: Number of dimensions for UMAP
        min_dist: Minimum distance for UMAP
        min_samples: Minimum samples for HDBSCAN
        min_cluster_size: Minimum cluster size for HDBSCAN
        cluster_selection_epsilon: Epsilon for cluster selection in HDBSCAN
        
    Returns:
        tuple: (umap_model, hdbscan_model)
    """
    logger.info("🚀 Using GPU-accelerated cuML UMAP for chunk processing")
    umap_model = cumlUMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric='cosine',
        random_state=42,
        output_type="numpy"  # Ensure NumPy output
    )
    
    logger.info("🚀 Using GPU-accelerated cuML HDBSCAN for chunk processing")
    hdbscan_model = cumlHDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_epsilon=cluster_selection_epsilon if cluster_selection_epsilon else 0.2  # Default value
    )
    return umap_model, hdbscan_model