"""
Module for topic modeling using BERTopic with optimizations for large datasets.
"""

from ast import Raise
import os
import time
import logging
import torch
import numpy as np
import traceback
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from eCallsAgent.utils.openai_compat import create_openai_client, create_completion
import json
import pickle
import gc
import pandas as pd
import csv
from eCallsAgent.core.embedding_generator import EmbeddingGenerator
from eCallsAgent.config import global_options as gl # global settings
import matplotlib.pyplot as plt
from .model_eval import ModelEvaluator
import sys  # Added for system debugging
from openai import OpenAI
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Union, Optional
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import random
import math
import tempfile
import shutil
from datetime import datetime
import collections
# Import necessary libraries for image saving
import plotly.io as pio
# Try to import kaleido which is required for image export
import kaleido
# Environment variable to control Numba CUDA usage
import os
import re
ENABLE_NUMBA_CUDA = os.environ.get('ENABLE_NUMBA_CUDA', 'True').lower() in ('true', '1', 't')
from eCallsAgent.core.visualization import TopicVis

# Configure Numba and CUDA logger levels
import logging
numba_logger = logging.getLogger('numba.cuda.cudadrv.driver')
numba_logger.setLevel(logging.CRITICAL)  # Suppress Numba CUDA driver errors
ptx_logger = logging.getLogger('ptxcompiler.patch')
ptx_logger.setLevel(logging.CRITICAL)  # Suppress ptxcompiler patch errors

# Import the cuda_setup utility
from eCallsAgent.utils.cuda_setup import setup_cuda, check_cuml_availability 

# Run CUDA setup before imports
CUDA_READY, CUDA_DEVICE, CUDA_MEMORY = setup_cuda()
CUML_AVAILABLE = check_cuml_availability() if CUDA_READY else False

try:
    import importlib.util
    import subprocess
    
    # Log system information
    logger = logging.getLogger(__name__)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"CUDA available via PyTorch: {torch.cuda.is_available()}")
    logger.info(f"CUDA initialized successfully: {CUDA_READY}")
    logger.info(f"cuML available: {CUML_AVAILABLE}")
    
    if CUDA_READY:
        logger.info(f"Using CUDA device: {CUDA_DEVICE} with {CUDA_MEMORY:.2f} GB memory")
except Exception as e:
    logger.error(f"Error in setup section: {e}")

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("CUDA is not available. Using CPU.")

class TopicModeler:
    """Wraps topic modeling components and handles training and saving of results."""
    def __init__(self, device: str):
        """
        Initialize the topic modeler.

        Args:
            device: The device to use for the topic modeler (cpu, cuda)
        """
        # Store the device specification
        self.device = device
        
        # Initialize parameters with defaults
        self.logger = logging.getLogger(__name__)
        
        # Import parameters from global options
        from eCallsAgent.config.global_options import (
            EMBEDDING_MODELS, DEFAULT_MODEL_INDEX, 
            N_NEIGHBORS, N_COMPONENTS, MIN_DIST,
            MIN_SAMPLES, MIN_CLUSTER_SIZE, 
            NR_TOPICS, METRIC, TOP_N_WORDS,
            MAX_DF, MIN_DF,
            MIN_DOCS_PER_TOPIC, MAX_DOCS_PER_TOPIC,
            MAX_ADAPTIVE_REPRESENTATIVES,
            SEED_TOPICS
        )
        
        # Set embedding model selection
        self.embedding_models = EMBEDDING_MODELS
        self.embedding_model_idx = DEFAULT_MODEL_INDEX
        self.embedding_model = self.embedding_models[self.embedding_model_idx]
        
        # Store parameters for UMAP 
        self.n_neighbors = N_NEIGHBORS
        self.n_components = N_COMPONENTS
        self.min_dist = MIN_DIST
        self.metric = METRIC
        
        # Store parameters for HDBSCAN
        self.min_samples = MIN_SAMPLES
        self.min_cluster_size = MIN_CLUSTER_SIZE
        
        # Store parameters for BERTopic
        self.nr_topics = NR_TOPICS
        self.top_n_words = TOP_N_WORDS
        
        # Store parameters for vectorizer
        self.max_df = MAX_DF
        self.min_df = MIN_DF
        
        # Store parameters for topic distillation
        self.min_docs_per_topic = MIN_DOCS_PER_TOPIC
        self.max_docs_per_topic = MAX_DOCS_PER_TOPIC
        self.max_adaptive_representatives = MAX_ADAPTIVE_REPRESENTATIVES
        
        # Store seed topics
        self.seed_topics = SEED_TOPICS
        
        # Initialize document storage for visualizations
        self.processed_documents = None
        self.document_topics = None
        
        # Initialize collections for representative documents
        self.representative_docs = []
        self.representative_embeddings = []
        
        # Setup folder structure and models
        self._setup_folder_structure()
        self._setup_openai()
        
        # Set cuML availability flag
        self.cuml_available = CUML_AVAILABLE
        
        # Set the embedding model to use
        self.embedding_model = gl.EMBEDDING_MODELS[gl.DEFAULT_MODEL_INDEX]
        logger.info(f"Using embedding model: {self.embedding_model}")
        
        # Set the number of topics
        self.n_topics = gl.NR_TOPICS[0]
        
        # Initialize model evaluator
        self.model_evaluator = ModelEvaluator()
        
        # REMOVED: Redundant initialization of output_dirs - already done in _setup_folder_structure

    def _setup_openai(self):
        """Set up OpenAI client for generating topic labels."""
        try:
            import openai
            
            # Read API key from file instead of environment variable
            api_key_file = os.path.join(gl.input_folder, 'raw', "OPENAI_API_KEY.txt")
            
            if not os.path.exists(api_key_file):
                self.logger.error(f"OpenAI API key file not found at: {api_key_file}")
                self.openai_available = False
                return
                
            try:
                # Read the API key from the file (strip to remove any whitespace/newlines)
                with open(api_key_file, 'r') as f:
                    api_key = f.read().strip()
                    
                if not api_key:
                    self.logger.error("OpenAI API key file is empty")
                    self.openai_available = False
                    return
                    
                # Create the client with the API key from file
                self.client = openai.OpenAI(api_key=api_key)
                
                # Test connection
                response = self.client.models.list()
                self.available_models = [model.id for model in response.data]
                self.logger.info(f"OpenAI connection successful. Available models: {self.available_models[:5]}...")
                self.openai_available = True
                
            except Exception as file_error:
                self.logger.error(f"Error reading API key from file: {file_error}")
                self.openai_available = False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.openai_available = False

    def _create_chunk_model(self, chunk_size: int = None) -> BERTopic:
        """Create optimized model for chunk processing"""
        # If chunk_size is not provided, calculate it
        if chunk_size is None:
            chunk_size = self._calculate_optimal_batch_size()
            logger.info(f"No chunk size provided, using calculated optimal size: {chunk_size}")
        
        # Use optimal parameters from grid search results
        n_neighbors = 5       # Grid search optimal value
        n_components = 50     # Grid search optimal value
        min_dist = 0.0        # Grid search optimal value
        min_samples = 5       # Grid search optimal value
        min_cluster_size = 30 # Grid search optimal value
        
        logger.info(f"Using grid search optimal parameters: n_neighbors={n_neighbors}, n_components={n_components}, min_dist={min_dist}")
        logger.info(f"HDBSCAN parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        logger.info(f"Creating chunk model with chunk size: {chunk_size}")
        
        return BERTopic(
            umap_model=UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                metric='cosine',
                random_state=42,
                verbose=False,
                low_memory=True,
                n_jobs=-1
            ),
            hdbscan_model=HDBSCAN(
                min_samples=min_samples,
                min_cluster_size=min_cluster_size,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True,
                core_dist_n_jobs=-1
            ),
            embedding_model=None,  # Set to None since we're using pre-computed embeddings
            calculate_probabilities=False,
            verbose=False
        )

    def _store_representatives(self, chunk_model, chunk_docs, chunk_embeddings, topic_representatives):
        """
        DEPRECATED: Use _store_representatives_fast instead.
        This method is kept for backward compatibility but will be removed in future versions.
        
        For new code, use _store_representatives_fast which has better performance.
        """
        # Forward to the fast implementation
        # Create an empty topic_to_docs dictionary
        topic_to_docs = {}
        for idx, topic in enumerate(chunk_model.topics_):
            if topic not in topic_to_docs:
                topic_to_docs[topic] = []
            topic_to_docs[topic].append(idx)
            
        return self._store_representatives_fast(chunk_model, chunk_docs, chunk_embeddings, topic_representatives, topic_to_docs)

    def _transform_documents_gpu(self, docs: list, embeddings: np.ndarray, chunk_size: int = 5000) -> list:
        """
        Transform documents using GPU acceleration for large datasets.
        
        Args:
            docs: List of documents to transform
            embeddings: Numpy array of embeddings corresponding to the documents
            chunk_size: Size of chunks to process at once
            
        Returns:
            list: Topic assignments for each document
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Using GPU-accelerated document transformation for {len(docs)} documents")
        
        try:
            # Determine available GPU resources
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                logger.warning("No GPUs available, falling back to CPU")
                # Attempt to transform with CPU
                topics, _ = self.topic_model.transform(docs, embeddings)
                return topics
            
            gpus_per_node = min(4, num_gpus)  # Use at most 4 GPUs per node
            logger.info(f"Using {gpus_per_node} GPUs for document transformation")
            
            # Initialize results list
            all_topics = []
            
            # Split data across GPUs
            docs_per_gpu = len(docs) // gpus_per_node
            remainder = len(docs) % gpus_per_node
            
            for gpu_id in range(gpus_per_node):
                # Calculate start and end indices for this GPU
                start_idx = gpu_id * docs_per_gpu + min(gpu_id, remainder)
                end_idx = start_idx + docs_per_gpu + (1 if gpu_id < remainder else 0)
                
                # Process data on this GPU
                logger.info(f"Processing documents {start_idx}-{end_idx} on GPU {gpu_id}")
                
                # Store results for this GPU
                gpu_results = []
                
                # Process in chunks to avoid OOM
                for chunk_start in tqdm(range(start_idx, end_idx, chunk_size),
                                      desc=f"Processing GPU {gpu_id}"):
                    chunk_end = min(chunk_start + chunk_size, end_idx)
                    
                    # Get chunk data
                    chunk_embeddings = embeddings[chunk_start:chunk_end]
                    
                    try:
                        with torch.cuda.device(f"cuda:{gpu_id % gpus_per_node}"):
                            # Transform embeddings using UMAP
                            umap_embeddings = self.topic_model.umap_model.transform(chunk_embeddings)
                            
                            # Fit HDBSCAN and get clusters
                            self.topic_model.hdbscan_model.fit(umap_embeddings)
                            labels = self.topic_model.hdbscan_model.labels_
                            
                            # Clear GPU memory
                            torch.cuda.empty_cache()
                            gpu_results.extend(labels)
                            
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_start}-{chunk_end} on GPU {gpu_id}: {str(e)}")
                        # Assign noise label (-1) to failed chunks
                        gpu_results.extend([-1] * (chunk_end - chunk_start))
                
                # Collect results from this GPU
                all_topics.extend(gpu_results)
                
                # Log progress
                logger.info(f"Completed processing on GPU {gpu_id}, processed {len(gpu_results)} documents")
            
            # Verify results
            if len(all_topics) == 0:
                logger.warning("No topics were generated. Returning default assignments.")
                return [-1] * len(docs)
                
            if len(all_topics) != len(docs):
                logger.warning(f"Mismatch in results: got {len(all_topics)} labels for {len(docs)} documents")
                # Pad with noise labels if necessary
                if len(all_topics) < len(docs):
                    all_topics.extend([-1] * (len(docs) - len(all_topics)))
                else:
                    # Trim if somehow we got more labels than documents
                    all_topics = all_topics[:len(docs)]
            
            logger.info(f"Completed distributed transformation of {len(all_topics)} documents")
            return all_topics
            
        except Exception as e:
            logger.error(f"Error in distributed document transformation: {e}")
            logger.error(traceback.format_exc())
            # Return noise labels for all documents in case of failure
            return [-1] * len(docs)

    def train_topic_model(self, docs: list, embeddings: list) -> BERTopic:
        """
        Train a topic model on the provided documents.
        
        Args:
            documents: List of documents to model
            embeddings: List of document embeddings
            save_visualizations: Whether to save visualizations (default: True)
            specific_visualizations: List of specific visualizations to save, options:
                                    ['barchart', 'hierarchy', 'heatmap', 'distance', 'embeddings']
                                    If None, all visualizations will be saved.
        
        Returns:
            BERTopic: Trained topic model
        """
        try:
            # Log the start of training
            start_time = time.time()
            self.logger.info(f"Starting topic modeling process with {len(docs)} documents")
            
            # Calculate adaptive parameters for earnings calls
            self.logger.info("**************Phase 0: Calculating adaptive parameters**************")
            adaptive_params = self._calculate_adaptive_parameters(docs, embeddings)
            self.logger.info(f"Using adaptive min_cluster_size: {adaptive_params['min_cluster_size']}, " + 
                            f"n_neighbors: {adaptive_params['n_neighbors']}, " +
                            f"n_components: {adaptive_params['n_components']}")
            
            # Preprocessing
            self.logger.info("**************Phase 1: Preprocessing data**************")
            preprocessed_docs, preprocessed_embeddings = self._preprocess_data(docs, embeddings)
            
            # Process the data in chunks to avoid memory issues
            self.logger.info("**************Phase 2: Processing data in chunks**************")
            topic_representatives = self._process_chunks(preprocessed_docs, preprocessed_embeddings)
            
            # Distill the topics to get representative documents
            self.logger.info("**************Phase 3: Distilling topics to get representative documents**************")
            all_rep_docs, all_rep_embeddings, topic_counts = self._distill_topics(topic_representatives, preprocessed_docs, preprocessed_embeddings)
            
            # Train the final model
            self.logger.info("**************Phase 4: Training final topic model on representative documents**************")
            topic_model = self._train_final_model(all_rep_docs, all_rep_embeddings)
            
            # Map all documents to topics
            self.logger.info("**************Phase 5: Mapping all documents to topics**************")
            topics, probs = self._map_documents(preprocessed_docs, preprocessed_embeddings)

            self.logger.info("**************Phase 6: Saving all visualizations**************")
            # Save all visualizations using the save_figures method
            self.save_figures(topic_model)
    
            # Clean up temporary files
            self._cleanup_temp_files()
            
            # Log completion
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.logger.info(f"**************Topic modeling completed in {elapsed_time:.2f} seconds**************")
            
            return self.topic_model
            
        except Exception as e:
            self.logger.error(f"Error training topic model: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def _preprocess_data(self, docs: list, embeddings: np.ndarray) -> tuple:
        """Preprocess input data with memory mapping."""
        docs = [str(doc) for doc in docs]
        
        # Create memory-mapped array for embeddings
        mmap_path = os.path.join(gl.output_folder, 'temp_embeddings.mmap')
        shape = embeddings.shape
        mmap_embeddings = np.memmap(mmap_path, dtype=np.float32, mode='w+', shape=shape)
        mmap_embeddings[:] = embeddings[:]
        
        return docs, mmap_embeddings

    def _process_chunks(self, docs: list, embeddings: np.ndarray) -> dict:
        """Process documents in chunks to identify potential topics."""
        try:
            # Initialize topic representatives dictionary
            topic_representatives = {}
            
            # Calculate optimal chunk size based on GPU memory
            chunk_size = self._calculate_optimal_batch_size()
            
            # Process documents in chunks
            for i in range(0, len(docs), chunk_size):
                chunk_docs = docs[i:i+chunk_size]
                chunk_embeddings = embeddings[i:i+chunk_size]
                
            self.logger.info(f"Processing documents in chunks of size {chunk_size}")
            
            # Create models once and reuse them for all chunks
            self.logger.info("Creating UMAP and HDBSCAN models for chunk processing")
            
            # Use optimal parameters from grid search results
            n_neighbors = 5       # Grid search optimal value
            n_components = 50     # Grid search optimal value
            min_dist = 0.0        # Grid search optimal value
            min_samples = 5       # Grid search optimal value
            min_cluster_size = 30 # Grid search optimal value
            
            # Create the models once
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
            
            # Create a single BERTopic model instance to reuse
            chunk_model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                embedding_model=None,  # Set to None since we're using pre-computed embeddings
                calculate_probabilities=False,
                verbose=False
            )
            
            logger.info(f"Using grid search optimal parameters: n_neighbors={n_neighbors}, n_components={n_components}, min_dist={min_dist}")
            logger.info(f"HDBSCAN parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
            
            # Process documents in chunks
            n_chunks = (len(docs) + chunk_size - 1) // chunk_size
            
            # Track retry statistics
            retry_needed = 0
            retry_successful = 0
            
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(docs))
                
                # Skip if no documents in this chunk
                if start_idx >= len(docs):
                    continue
                
                # Get chunk of documents and embeddings
                chunk_docs = docs[start_idx:end_idx]
                chunk_embeddings = embeddings[start_idx:end_idx]
                
                logger.info(f"Processing chunk {i+1}/{n_chunks} with {len(chunk_docs)} documents")
                logger.info(f"Chunk embeddings shape: {chunk_embeddings.shape}")
                
                # Fit chunk model without recreating it
                try:
                    # Fit UMAP
                    transformed_embeddings = umap_model.fit_transform(chunk_embeddings)
                    
                    # Fit HDBSCAN on transformed embeddings
                    hdbscan_model.fit(transformed_embeddings)
                    
                    # Get cluster assignments
                    topics = hdbscan_model.labels_
                    
                    # Update the model's internal state
                    chunk_model.topics_ = topics
                    
                    # Get number of topics
                    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
                    
                    logger.info(f"Found {n_topics} topics in chunk {i+1}")
                    
                    # If we find very few topics, retry with more aggressive parameters
                    if n_topics < 5:
                        retry_needed += 1
                        logger.warning(f"Few topics found ({n_topics}), retrying with more aggressive parameters")
                        
                        # Create a more aggressive model for retry
                        retry_umap = UMAP(
                            n_neighbors=5,            # Keep optimal value
                            n_components=75,          # Increase from optimal 50
                            min_dist=0.0,            # Keep optimal value
                            metric='cosine',
                            random_state=42,
                            verbose=False,
                            low_memory=True
                        )
                        
                        retry_hdbscan = HDBSCAN(
                            min_samples=3,           # Reduce from optimal 5
                            min_cluster_size=15,     # Reduce from optimal 30
                            metric='euclidean',
                            cluster_selection_method='eom',
                            prediction_data=True
                        )
                        
                        # Apply more aggressive model
                        retry_transformed = retry_umap.fit_transform(chunk_embeddings)
                        retry_hdbscan.fit(retry_transformed)
                        retry_topics = retry_hdbscan.labels_
                        
                        # Get number of topics with retry model
                        retry_n_topics = len(set(retry_topics)) - (1 if -1 in retry_topics else 0)
                        
                        logger.info(f"Retry found {retry_n_topics} topics (was {n_topics})")
                        
                        # Use the result with more topics
                        if retry_n_topics > n_topics:
                            retry_successful += 1
                            topics = retry_topics
                            transformed_embeddings = retry_transformed
                            n_topics = retry_n_topics
                            chunk_model.topics_ = topics
                            logger.info(f"Using retry result with {n_topics} topics")
                    
                    # Create a topic-to-document mapping for representative docs
                    topic_to_docs = {}
                    for doc_idx, topic_label in enumerate(topics):
                        if topic_label not in topic_to_docs:
                            topic_to_docs[topic_label] = []
                        topic_to_docs[topic_label].append(doc_idx)
                    
                    # Store representative documents
                    self._store_representatives_fast(chunk_model, chunk_docs, chunk_embeddings, topic_representatives, 
                                                   topic_to_docs)
                except Exception as chunk_error:
                    logger.error(f"Error processing chunk {i+1}: {chunk_error}")
                    logger.error(traceback.format_exc())
                    continue
                
                # Clear memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Debug topic representatives structure
                if i == 0:
                    self._debug_topic_representatives(topic_representatives)
            
            # Save topic representatives to file
            representatives_path = os.path.join(self.output_dirs['models'], 'topic_representatives.json')
            with open(representatives_path, 'w', encoding='utf-8') as f:
                json.dump(topic_representatives, f, ensure_ascii=False, indent=2)
            
            # Log final statistics
            total_topics = len(topic_representatives)
            total_docs = sum(len(topic_data.get('docs', [])) for topic_data in topic_representatives.values())
            logger.info(f"Completed chunk processing with {total_topics} total topics and {total_docs} representative documents")
            logger.info(f"Retry statistics: {retry_needed} chunks needed retry, {retry_successful} were successful")
            
            return topic_representatives
            
        except Exception as e:
            logger.error(f"Error in _process_chunks: {e}")
            logger.error(traceback.format_exc())
            raise

    def _distill_topics(self, topic_representatives: dict, docs: list, embeddings: np.ndarray) -> tuple:
        """Distill topics from chunks to get representative documents."""
        try:
            # Determine the maximum number of representative documents
            # This needs to be large enough to provide good coverage, but small enough to fit in memory
            max_representatives = self._calculate_max_representatives(len(docs))
            
            # FIXED: Increase minimum and maximum docs per topic to ensure enough representation
            min_docs_per_topic = gl.MIN_DOCS_PER_TOPIC  # Ensure at least 10 docs per topic
            max_docs_per_topic = gl.MAX_DOCS_PER_TOPIC  # Ensure at most 300 max docs
            
            # Set target number of topics
            target_topics = gl.NR_TOPICS[0]  # Set target to 300 from global options
            
            # Log parameters for transparency
            logger.info(f" Distilling topics with parameters:")
            logger.info(f"  - Max representatives: {max_representatives}")
            logger.info(f"  - Min docs per topic: {min_docs_per_topic}")
            logger.info(f"  - Max docs per topic: {max_docs_per_topic}")
            logger.info(f"  - Target topics: {target_topics}")
            
            # Get the sorted list of topics
            sorted_topics = sorted(topic_representatives.keys())
            logger.info(f"Sorted {len(sorted_topics)} topics")
            
            # IMPROVED: Use an adaptive approach for target topics based on corpus size
            if len(docs) > 500000:
                # Very large corpus - up to 6000 topics
                target_topics = min(1000, len(sorted_topics))
                logger.info(f"Very large corpus: using up to {target_topics} topics")
            elif len(docs) > 100000:
                # Large corpus - up to 4000 topics
                target_topics = min(800, len(sorted_topics))
                logger.info(f"Large corpus: using up to {target_topics} topics")
            elif len(docs) > 50000:
                # Medium corpus - up to 3000 topics
                target_topics = min(500, len(sorted_topics))
                logger.info(f"Medium corpus: using up to {target_topics} topics")
            else:
                # Smaller corpus - up to 2000 topics
                target_topics = min(2000, len(sorted_topics))
                logger.info(f"Smaller corpus: using up to {target_topics} topics")
            
            # IMPROVED: Increase max docs per topic for larger topics and decrease for smaller topics
            # This creates more diversity in doc counts per topic
            topic_sizes = [len(topic_representatives[t].get('docs', [])) for t in sorted_topics[:target_topics]]
            
            # Use different max settings for different topic size quantiles
            if topic_sizes:
                # Calculate size quartiles (divide topics into 4 groups by size)
                q1 = np.percentile(topic_sizes, 25)
                q2 = np.percentile(topic_sizes, 50)
                q3 = np.percentile(topic_sizes, 75)
                
                # Log the quartiles for debugging
                logger.info(f"Topic size quartiles: Q1={q1:.1f}, Q2={q2:.1f}, Q3={q3:.1f}")
                
                # Use different max documents per topic based on corpus size and topic size distribution
                # Significantly increased values to target ~100 docs per topic on average
                max_docs_ranges = {
                    "largest": min(200, max(150, len(docs) // 500)),    # 75-100 percentile
                    "large": min(150, max(120, len(docs) // 800)),      # 50-75 percentile
                    "medium": min(100, max(100, len(docs) // 1000)),    # 25-50 percentile
                    "small": min(50, max(80, len(docs) // 1500))       # 0-25 percentile
                }
                
                logger.info(f"Using adaptive max docs per topic: {max_docs_ranges}")
                
                # Create a function to determine max_docs based on topic size
                def get_max_docs(topic_size):
                    if topic_size >= q3:
                        return max_docs_ranges["largest"]
                    elif topic_size >= q2:
                        return max_docs_ranges["large"]
                    elif topic_size >= q1:
                        return max_docs_ranges["medium"]
                    else:
                        return max_docs_ranges["small"]
                
                # Collect representative documents, with varying max_docs_per_topic per topic
                all_rep_docs = []
                all_rep_embeddings = []
                topic_counts = {}
                
                # Process each topic with its own max_docs limit
                for topic in sorted_topics[:target_topics]:
                    topic_data = topic_representatives[topic]
                    topic_size = len(topic_data.get('docs', []))
                    max_docs = get_max_docs(topic_size)
                    
                    # Get representative documents for this topic
                    topic_rep_docs, topic_rep_embeddings, topic_count = self._collect_representative_for_topic(
                        topic, topic_data, 
                        min_docs=gl.MIN_DOCS_PER_TOPIC, 
                        max_docs=max_docs,
                        max_representatives=max_representatives
                    )
                    
                    all_rep_docs.extend(topic_rep_docs)
                    all_rep_embeddings.extend(topic_rep_embeddings)
                    topic_counts[topic] = topic_count
                    
                    # Log topic information
                    logger.info(f"Topic {topic} has {topic_count} docs (max allowed: {max_docs}, original size: {topic_size})")
                    
                # Log overall statistics
                if topic_counts:
                    avg_docs = sum(topic_counts.values()) / len(topic_counts)
                    min_docs = min(topic_counts.values())
                    max_docs = max(topic_counts.values())
                    logger.info(f"Topics processed: {len(topic_counts)}, "
                                f"Avg docs/topic: {avg_docs:.1f}, "
                                f"Min: {min_docs}, Max: {max_docs}")
                
                return all_rep_docs, all_rep_embeddings, topic_counts
            else:
                # Fallback to original method if no topic sizes
                max_docs_per_topic = min(300, max(100, len(docs) // 800))  # Increased from 200 to 300 with minimum 100
                logger.info(f"Using fixed max_docs_per_topic={max_docs_per_topic} based on corpus size")
            
            # Collect representative documents, up to the maximum
            all_rep_docs, all_rep_embeddings, topic_counts = self._collect_representatives(
                sorted_topics[:target_topics],
                max_representatives,
                gl.MIN_DOCS_PER_TOPIC,
                max_docs_per_topic
            )
            
            return all_rep_docs, all_rep_embeddings, topic_counts
        except Exception as e:
            logger.error(f"Error in distill_topics: {e}")
            logger.error(traceback.format_exc())
            raise

    def _collect_representative_for_topic(self, topic, topic_data, min_docs, max_docs, max_representatives):
        """
        Collect representative documents for a specific topic.
        
        Args:
            topic: Topic identifier
            topic_data: Dictionary containing documents and embeddings for the topic
            min_docs: Minimum number of documents to collect
            max_docs: Maximum number of documents to collect
            max_representatives: Overall maximum representatives constraint
            
        Returns:
            tuple: (documents, embeddings, count)
        """
        docs = topic_data.get('docs', [])
        embeddings = topic_data.get('embeddings', [])
        
        # Ensure we have the same number of docs and embeddings
        if len(docs) != len(embeddings):
            logger.warning(f"Topic {topic} has mismatched docs ({len(docs)}) and embeddings ({len(embeddings)})")
            # Take the minimum of the two to avoid errors
            min_length = min(len(docs), len(embeddings))
            docs = docs[:min_length]
            embeddings = embeddings[:min_length]
        
        # Determine how many documents to collect (between min and max)
        num_docs = max(min_docs, min(max_docs, len(docs)))
        
        # Take a random sample if we have more than enough documents
        if len(docs) > num_docs:
            # Get random indices without replacement
            indices = random.sample(range(len(docs)), num_docs)
            sampled_docs = [docs[i] for i in indices]
            sampled_embeddings = [embeddings[i] for i in indices]
        else:
            # Use all documents if we have fewer than num_docs
            sampled_docs = docs
            sampled_embeddings = embeddings
        
        return sampled_docs, sampled_embeddings, len(sampled_docs)

    def _train_final_model(self, all_rep_docs: list, all_rep_embeddings: list) -> None:
        """
        Train the final topic model on representative documents.
        
        Args:
            all_rep_docs: List of representative documents
            all_rep_embeddings: List of representative embeddings
        
        Returns:
            BERTopic: Trained topic model
        """
        try:
            self.logger.info(f"Training final model on {len(all_rep_docs)} representative documents")
            
            # CRITICAL FIX: Convert list of embeddings to a single numpy array
            if not isinstance(all_rep_embeddings, np.ndarray):
                logger.info(f"Converting embeddings from list to numpy array")
                all_rep_embeddings = np.array(all_rep_embeddings, dtype=np.float32)
                logger.info(f"Embeddings converted, shape: {all_rep_embeddings.shape}")
            
            # Use adaptive parameters if available
            if hasattr(self, 'adaptive_parameters'):
                # Use calculated adaptive parameters
                n_neighbors = self.adaptive_parameters["n_neighbors"]
                n_components = self.adaptive_parameters["n_components"]
                min_dist = self.adaptive_parameters["min_dist"]
                min_cluster_size = self.adaptive_parameters["min_cluster_size"]
                min_samples = self.adaptive_parameters["min_samples"]
                cluster_selection_epsilon = self.adaptive_parameters["cluster_selection_epsilon"]
                
                logger.info(f"Using adaptive parameters for earnings call analysis:")
            else:
                # Fallback to optimized default parameters if adaptive parameters aren't available
                n_neighbors = gl.N_NEIGHBORS[0] if hasattr(gl, 'N_NEIGHBORS') and gl.N_NEIGHBORS else 15
                n_components = gl.N_COMPONENTS[0] if hasattr(gl, 'N_COMPONENTS') and gl.N_COMPONENTS else 5
                min_dist = 0.1
                min_cluster_size = gl.MIN_CLUSTER_SIZE[0] if hasattr(gl, 'MIN_CLUSTER_SIZE') and gl.MIN_CLUSTER_SIZE else 10
                min_samples = max(3, min_cluster_size // 5)
                cluster_selection_epsilon = 0.2
                
                logger.info(f"Using default parameters (adaptive parameters not available):")
            
            logger.info(f"  - n_neighbors: {n_neighbors}")
            logger.info(f"  - n_components: {n_components}")
            logger.info(f"  - min_dist: {min_dist}")
            logger.info(f"  - min_cluster_size: {min_cluster_size}")
            logger.info(f"  - min_samples: {min_samples}")
            logger.info(f"  - cluster_selection_epsilon: {cluster_selection_epsilon}")
            
            # Initialize UMAP with optimized parameters
            umap_model = UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                metric='cosine',
                low_memory=True,
                random_state=42
            )
            
            # Initialize HDBSCAN with optimized parameters
            hdbscan_model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_epsilon=cluster_selection_epsilon,
                gen_min_span_tree=True,
                prediction_data=True
            )
            
            # Create the BERTopic model with the optimized models
            topic_model = self._create_topic_model(umap_model, hdbscan_model)
            
            # Set the number of topics if specified
            if hasattr(self, 'n_topics') and self.n_topics != "auto" and isinstance(self.n_topics, int):
                logger.info(f"Setting number of topics to {self.n_topics}")
                topic_model.verbose = True
                topic_model = topic_model.update_topics(all_rep_docs, n_topics=self.n_topics)
            
            # Fit model with more verbosity
            logger.info("Fitting topic model on representative documents...")
            self.topic_model = topic_model
            self.topic_model.verbose = True
            
            # Fit the model to the representative documents
            try:
                self.topic_model.fit(all_rep_docs, all_rep_embeddings)
                
                # Get the number of topics
                topic_info = self.topic_model.get_topic_info()
                pre_reduction_topics = len(topic_info[topic_info.Topic != -1])
                logger.info(f"Initial model has {pre_reduction_topics} topics")
                
                # Reduce topics if needed
                if hasattr(self, 'n_topics') and isinstance(self.n_topics, int) and pre_reduction_topics > self.n_topics * 1.5:
                    logger.info(f"Reducing topics from {pre_reduction_topics} to target {self.n_topics}")
                    self._reduce_topics(all_rep_docs)
                
                # Print topic information
                topic_info = self.topic_model.get_topic_info()
                n_topics = len(topic_info[topic_info.Topic != -1])
                logger.info(f"Final model has {n_topics} topics")
                logger.info(f"Top 5 largest topics: {topic_info.iloc[:5].to_dict(orient='records')}")
                
                # Store the representative docs in the model
                # This will be useful for visualization later
                if not hasattr(self.topic_model, 'representative_docs_'):
                    self.topic_model.representative_docs_ = {}
                
                # Store representative documents for each topic
                for topic in set(self.topic_model.topics_):
                    if topic != -1:  # Skip outlier topic
                        doc_indices = [i for i, t in enumerate(self.topic_model.topics_) if t == topic]
                        topic_docs = [all_rep_docs[i] for i in doc_indices]
                        # Store up to 50 docs per topic to save memory
                        self.topic_model.representative_docs_[topic] = topic_docs[:50]
                
                return self.topic_model
                
            except Exception as e:
                logger.error(f"Error in topic modeling: {e}")
                logger.error(traceback.format_exc())
                raise
        except Exception as e:
            logger.error(f"Error in _train_final_model: {e}")
            logger.error(traceback.format_exc())
            raise

    def _map_documents(self, docs: list, embeddings: np.ndarray) -> tuple:
        """
        Map all documents to topics using the trained model.
        
        Args:
            docs: List of documents to map
            embeddings: Numpy array of embeddings corresponding to the documents
            
        Returns:
            tuple: (topics, probabilities) - topic assignments and their probabilities
        """
        try:
            self.logger.info(f"Mapping {len(docs)} documents to topics...")
            
            # Get the current topic model
            topic_model = self.topic_model
            
            # Check if the model has been trained
            if not hasattr(topic_model, 'umap_model') or not hasattr(topic_model, 'hdbscan_model'):
                self.logger.error("Topic model has not been properly trained yet")
                # Return default values instead of None
                return [-1] * len(docs), [[0.0]] * len(docs)
            
            # Use the GPU if available for faster processing
            if self.device == 'cuda' and torch.cuda.is_available():
                self.logger.info("Using GPU for document mapping")
                topics = self._transform_documents_gpu(docs, embeddings)
                
                # If topics is None or empty, use default assignment
                if topics is None or len(topics) == 0:
                    self.logger.warning("GPU transformation returned no topics, using default assignments")
                    topics = [-1] * len(docs)
                    
                # For GPU transformation, we might not have probabilities
                # Assign uniform probability to the assigned topic
                probs = [[1.0 if topic != -1 else 0.0] for topic in topics]
                
                return topics, probs
            else:
                # Use CPU for transformation
                self.logger.info("Using CPU for document mapping")
                
                # Transform into topic space
                topics, probs = topic_model.transform(docs, embeddings)
                
                return topics, probs
                
        except Exception as e:
            self.logger.error(f"Error mapping documents to topics: {e}")
            self.logger.error(traceback.format_exc())
            
            # Return default topic assignments and probabilities on error
            self.logger.warning("Returning default topic assignments due to error")
            return [-1] * len(docs), [[0.0]] * len(docs)

    def _calculate_max_representatives(self, total_docs):
        """
        Calculate the maximum number of representative documents to use based on the total number of documents.
        
        Args:
            total_docs: Either the total number of documents (as an integer) or the document collection itself.
            
        Returns:
            int: Maximum number of representative documents to use.
        """
        # Check if total_docs is already an integer
        if isinstance(total_docs, int):
            n_docs = total_docs
        else:
            # If it's a collection, get its length
            n_docs = len(total_docs)
        
        # For larger corpora, use a much more generous allocation to get 100 docs per topic
        # For 500 topics × 100 docs each, we need at least 50,000 docs total
        if n_docs > 1000000:
            # For large corpora, allocate 50% instead of 10%
            base_allocation = n_docs // 2  # Significantly increased from n_docs // 10
        else:
            # For smaller corpora, allocate 100%
            base_allocation = n_docs
        
        # Much higher minimum to ensure we get around 100 docs per topic
        return min(gl.MAX_ADAPTIVE_REPRESENTATIVES, base_allocation)

    def _collect_representatives(self, sorted_topics: list, max_representatives: int,
                               min_docs_per_topic: int, max_docs_per_topic: int) -> tuple:
        """
        Collect representative documents for each topic.
        
        Args:
            sorted_topics: Sorted list of topics by document count
            max_representatives: Maximum number of representative documents to collect
            min_docs_per_topic: Minimum number of documents to collect per topic
            max_docs_per_topic: Maximum number of documents to collect per topic
            
        Returns:
            tuple: (all_rep_docs, all_rep_embeddings, topic_counts)
        """
        try:
            all_rep_docs = []
            all_rep_embeddings = []
            topic_counts = {}
            
            # Calculate adaptive documents per topic based on topic sizes
            total_docs = sum(len(self.topic_representatives[topic]['docs']) for topic in sorted_topics)
            
            # Get the total count of documents across all topics
            self.logger.info(f"Number of topics to process: {len(sorted_topics)}")
            
            # FIXED: Override the target docs per topic to force at least 30 docs per topic
            # instead of dividing max_representatives evenly across topics
            min_desired_docs_per_topic = gl.MIN_DOCS_PER_TOPIC  # Changed from 20 to 30
            
            # Ensure we have enough docs allocated in total
            required_docs = len(sorted_topics) * min_desired_docs_per_topic
            if max_representatives < required_docs:
                self.logger.warning(f"Increasing max_representatives from {max_representatives} to {required_docs} to ensure at least {min_desired_docs_per_topic} docs per topic")
                max_representatives = required_docs
            
            # Calculate target docs per topic - fixed at 30 docs per topic
            target_docs_per_topic = {}
            remaining_docs = max_representatives
            
            # First pass: Set exactly 30 docs for each topic
            for topic in sorted_topics:
                # FIXED: Set exactly 30 docs per topic
                docs_for_topic = min_desired_docs_per_topic
                
                target_docs_per_topic[topic] = docs_for_topic
                remaining_docs -= docs_for_topic
                
            # Second pass: Collect documents based on the target counts
            for topic in sorted_topics:
                if topic not in self.topic_representatives:
                    continue
                    
                topic_data = self.topic_representatives[topic]
                rep_docs = topic_data.get('docs', [])
                rep_embeddings = topic_data.get('embeddings', [])
                
                # Ensure we have the same number of docs and embeddings
                min_length = min(len(rep_docs), len(rep_embeddings))
                if min_length < len(rep_docs) or min_length < len(rep_embeddings):
                    self.logger.warning(f"Topic {topic} has mismatched docs ({len(rep_docs)}) and embeddings ({len(rep_embeddings)})")
                    rep_docs = rep_docs[:min_length]
                    rep_embeddings = rep_embeddings[:min_length]
                
                # Determine how many documents to collect for this topic
                target_docs = target_docs_per_topic.get(topic, min_docs_per_topic)
                num_docs = min(target_docs, len(rep_docs))
                
                # Sample documents if we have more than needed
                if len(rep_docs) > num_docs:
                    # Get random indices without replacement
                    indices = random.sample(range(len(rep_docs)), num_docs)
                    sampled_docs = [rep_docs[i] for i in indices]
                    sampled_embeddings = [rep_embeddings[i] for i in indices]
                else:
                    # Use all documents if we have fewer than num_docs
                    sampled_docs = rep_docs
                    sampled_embeddings = rep_embeddings
                
                # Add to the collection
                all_rep_docs.extend(sampled_docs)
                all_rep_embeddings.extend(sampled_embeddings)
                topic_counts[topic] = len(sampled_docs)
            
            # Log the total and average counts
            if topic_counts:
                avg_docs = sum(topic_counts.values()) / len(topic_counts)
                min_docs = min(topic_counts.values())
                max_docs = max(topic_counts.values())
                self.logger.info(f"Topics processed: {len(topic_counts)}, "
                            f"Avg docs/topic: {avg_docs:.1f}, "
                            f"Min: {min_docs}, Max: {max_docs}")
            
            return all_rep_docs, all_rep_embeddings, topic_counts
            
        except Exception as e:
            self.logger.error(f"Error collecting representatives: {e}")
            self.logger.error(traceback.format_exc())
            # Return empty results to avoid breaking the pipeline
            return [], [], {}
    
    def _cleanup_temp_files(self):
        """Clean up temporary memory-mapped files."""
        try:
            temp_files = [
                # Add paths to any temporary files that need cleaning
                os.path.join(gl.output_folder, 'temp', f'{gl.data_filename_prefix}_embeddings.mmap'),
                os.path.join(gl.output_folder, 'temp', f'{gl.data_filename_prefix}_topic_keywords.pkl'),
                os.path.join(gl.output_folder, 'temp', f'{gl.data_filename_prefix}_topic_labels.json')
            ]
            
            for file_path in temp_files:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        self.logger.info(f"Removed temporary file: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"Could not remove temporary file {file_path}: {e}")
            
            # Clean up any large variables that are no longer needed
            if hasattr(self, 'topic_representatives'):
                del self.topic_representatives
                self.logger.info("Cleared topic_representatives from memory")
            
            # Run garbage collection to free memory
            import gc
            gc.collect()
            self.logger.info("Garbage collection completed")
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
            # Don't raise the exception - cleanup should not stop execution

    def generate_topic_label(self, keywords: list, docs: list) -> tuple:
        """Generate a meaningful label for a topic based on its keywords and representative documents.
        
        Args:
            keywords: List of keywords associated with the topic
            docs: List of representative documents for the topic
            
        Returns:
            tuple: (label, description, label_source)
        """
        try:
            # Ensure keywords list is valid
            if not keywords or not isinstance(keywords, list):
                self.logger.warning("Empty or invalid keywords list provided to generate_topic_label")
                return "Unlabeled Topic", "General"
                
            if not isinstance(docs, list):
                try:
                    docs = docs.tolist()
                except Exception as e:
                    docs = list(docs)

            # Check if docs is empty
            if docs is None or len(docs) == 0:
                top_words = [word for word, _ in keywords[:3] if isinstance(word, str)]
                label = " + ".join(top_words)
                return label, f"Based on keywords: {', '.join([w for w, _ in keywords[:5] if isinstance(w, str)])}", "keywords"            
                
            # Process keywords to handle different formats
            processed_keywords = []
            for k in keywords[:5]:
                # Handle tuples from BERTopic
                if isinstance(k, tuple) and len(k) >= 1:
                    processed_keywords.append(str(k[0]).replace('_', ' '))
                else:
                    processed_keywords.append(str(k).replace('_', ' '))
            
            self.logger.info(f"Generating label for keywords: {processed_keywords}")
            
            # Add caching to avoid redundant API calls
            cache_key = '_'.join(processed_keywords)  # Create a unique key
            cache_file = os.path.join(gl.output_folder, 'temp', 'topic_label_cache.json')
            
            # Check cache first and return cached value if available
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cache = json.load(f)
                        if cache_key in cache:
                            cached_value = cache[cache_key]
                            # Ensure we return exactly two values
                            if isinstance(cached_value, list) and len(cached_value) >= 2:
                                return cached_value[0].strip(), cached_value[1].strip()
                            elif isinstance(cached_value, str) and "," in cached_value:
                                parts = cached_value.split(",", 1)
                                return parts[0].strip(), parts[1].strip()
                            elif isinstance(cached_value, str):
                                return cached_value.strip(), "General"
                except Exception as e:
                    self.logger.warning(f"Error reading from cache: {e}")
            
            # Check if OpenAI is available; if not, use a simple fallback
            if not getattr(self, 'openai_available', False) or not hasattr(self, 'client'):
                self.logger.warning("OpenAI not available. Using fallback labeling.")
                main_topic = processed_keywords[0].title() if processed_keywords else "Unlabeled Topic"
                subtopic = ' '.join(processed_keywords[1:3]).title() if len(processed_keywords) > 1 else "General"
                return main_topic, subtopic
            
            # Prepare prompt for OpenAI
            doc_context = ""
            if docs and len(docs) > 0:
                doc_context = "\nExample discussions:\n" + "\n".join(docs[:2])
            prompt = f"""Generate a concise business topic label (1-2 words) and a subtopic label (2-3 words) based on the provided earnings call keywords.
                ### Examples:
                - **Topic**: "Business Strategy", **Subtopic**: "Market Expansion", "Mergers & Acquisitions", "Product Development", "Cost Optimization", "Others"
                - **Topic**: "Financial Position", **Subtopic**: "Debt Management", "Liquidity Risk", "Cash Flow", "Working Capital", "Others"
                - **Topic**: "Corporate Governance", **Subtopic**: "Board Structure", "Executive Compensation", "Regulatory Compliance", "Others"   
                - **Topic**: "Technology & Innovation", **Subtopic**: "Artificial Intelligence", "Digital Transformation", "R&D Investment", "Others"
                - **Topic**: "Risk Management", **Subtopic**: "Market Risk", "Operational Risk", "Regulatory Uncertainty", "Financial Stability", "Others"
                - **Topic**: "Market", **Subtopic**: "Market Expansion", "Mergers & Acquisitions", "Product Development", "Cost Optimization", "Others"
                - **Topic**: "Business Overview", **Subtopic**: "Business Strategy", "Company Description", "Geographic Presence", "Industry Trends", "Market Position", "Product Offerings", "Others"
                - **Topic**: "Contractual Obligations", **Subtopic**: "Revenue", "Earnings Per Share", "Gross Margin", "Net Income", "Others"
                - **Topic**: "Critical Accounting Policies", **Subtopic**: "Allowance for Doubtful Accounts", "Goodwill Impairment", "Income Taxes", "Inventory Valuation", "Revenue Recognition", "Share-Based Compensation", "Others"
                - **Topic**: "Financial Performance", **Subtopic**: "EBITDA", "Earnings Per Share", "Expenses", "Gross Profit", "Net Income", "Operating Income", "Revenues", "Others"
                - **Topic**: "Forward Looking Statements", **Subtopic**: "Assumptions", "Future Outlook", "Growth Strategy", "Market Opportunities", "Potential Risks", "Projections", "Others"
                - **Topic**: "Liquidity and Capital Resources", **Subtopic**: "Capital Expenditures", "Cash Flow", "Credit Facilities", "Debt Management", "Financing Activities", "Investing Activities", "Working Capital", "Others"
                - **Topic**: "Off Balance Sheet Arrangements", **Subtopic**: "Commitments", "Contingent Liabilities", "Guarantees", "Leases", "Variable Interest Entities", "Others"
                - **Topic**: "Recent Accounting Pronouncements", **Subtopic**: "Adoption Impact", "Impact Assessment", "Implementation Plans", "New Standards", "Others"
                - **Topic**: "Recent Developments", **Subtopic**: "Acquisitions", "Divestitures", "New Products", "Strategic Initiatives", "Others"
                - **Topic**: "Regulatory and Legal Matters", **Subtopic**: "Compliance", "Environmental Compliance", "Legal Proceedings", "Legislative Changes", "Regulatory Changes", "Others"
                - **Topic**: "Risk_Factors", **Subtopic**: "Competitive Risks", "Economic Conditions", "Financial Risks", "Market Risks", "Operational Risks", "Regulatory Risks", "Others"
                - **Topic**: "Segment Information", **Subtopic**: "Geographic Segments", "Product Segments", "Customer Segments", "Segment Performance", "Segment Profitability", "Segment Revenue", "Others"
                - **Topic**: "Sustainability_and_CSR", **Subtopic**: "Environmental Impact", "Social Responsibility", "Sustainability Initiatives", "Others"
                - **Topic**: "Accounting Policies", **Subtopic**: "Amortization", "Depreciation", "Revenue Recognition", "Income Taxes", "Leases", "Fair Value", "Goodwill"
                - **Topic**: "Auditor Report", **Subtopic**: "Audit Opinion", "Critical Audit Matters", "Internal Controls", "Basis for Opinion"
                - **Topic**: "Cash Flow", **Subtopic**: "Operating Activities", "Investing Activities", "Financing Activities"
                - **Topic**: "Corporate Governance", **Subtopic**: "Board Structure", "Executive Compensation", "Internal Controls", "Strategic Planning"
                - **Topic**: "Financial Performance", **Subtopic**: "Revenue", "Operating Income", "Net Income", "EPS", "Segment Results"
                - **Topic**: "Financial Position", **Subtopic**: "Assets", "Liabilities", "Equity", "Working Capital", "Investments"
                - **Topic**: "Business Overview", **Subtopic**: "Business Model", "Market Position", "Geographic Presence", "Industry Overview"
                - **Topic**: "Competition", **Subtopic**: "Market Share", "Competitive Advantages", "Industry Trends"
                - **Topic**: "Environmental Risks", **Subtopic**: "Climate Change", "Sustainability", "Resource Management"
                - **Topic**: "External Factors", **Subtopic**: "Economic Conditions", "Geopolitical Risks", "Market Conditions"
                - **Topic**: "Financial Risks", **Subtopic**: "Credit Risk", "Liquidity Risk", "Interest Rate Risk", "Market Risk"
                - **Topic**: "Regulatory Matters", **Subtopic**: "Compliance", "Legal Proceedings", "Regulatory Changes"
                - **Topic**: "Strategic Initiatives", **Subtopic**: "Growth Strategy", "Market Expansion", "Innovation"
                - **Topic**: "Operational Performance", **Subtopic**: "Efficiency", "Productivity", "Cost Management"
                - **Topic**: "Market Analysis", **Subtopic**: "Market Trends", "Consumer Behavior", "Competition"
                - **Topic**: "Industry Specific Information", **Subtopic**: "Industry Policy", "Industry Trends", "Regulatory Environment", "Competitive Landscape", "Others"
                ### Keywords:
                - **Primary Keywords**: {', '.join(processed_keywords)}
                - **Secondary Keywords**: {', '.join([k.replace('_', ' ') for k in keywords[5:8] if 5 < len(keywords)])}

                ### Context:
                {doc_context}

                ### Requirements:
                - Use standard financial and business terminology
                - Ensure topic labels are broad yet meaningful
                - Ensure subtopics are specific and relevant
                - Classify topics based on:
                  1. Financial Fundamentals (performance, position, cash flow)
                  2. Business Operations (strategy, market, competition)
                  3. Governance & Control (policies, audits, compliance)
                  4. Risk Factors (financial, operational, external)
                - Output format: Topic: [Label], Subtopic: [Specific Area]
                - Be concise and specific."""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in firms' fundamental topic analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )

            if response and response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                
                # Extract label and description using regex
                label_match = re.search(r'Topic:(.*?)(?:\n|$)', content)
                desc_match = re.search(r'Subtopic:(.*?)(?:\n|$)', content)
                
                if label_match and desc_match:
                    label = label_match.group(1).strip()
                    description = desc_match.group(1).strip()
                    
                    # Store in cache
                    if not hasattr(self, 'label_cache'):
                        self.label_cache = {}
                    self.label_cache[cache_key] = (label, description, "openai")
                    return label, description, "openai"            

            # Fallback: Use NLP to extract key phrases from documents
            try:
                from sklearn.feature_extraction.text import CountVectorizer
                
                # Combine docs into a single text
                text = " ".join(docs)
                
                # Extract bigrams and trigrams
                n_gram_range = (2, 3)
                count_vectorizer = CountVectorizer(ngram_range=n_gram_range, stop_words='english')
                
                # Fit on the text
                X = count_vectorizer.fit_transform([text])
                
                # Get feature names and their counts
                features = count_vectorizer.get_feature_names_out()
                counts = X.toarray().sum(axis=0)
                
                # Initialize clean_keywords from the original keywords
                clean_keywords = []
                for word_item in keywords[:5]:
                    if isinstance(word_item, tuple) and len(word_item) >= 2:
                        word, weight = word_item
                    else:
                        # Handle case where keywords aren't (word, weight) tuples
                        word = str(word_item)
                        weight = 1.0
                    
                    # Skip if word is just a number or single character
                    if word.isdigit() or len(word) <= 1:
                        continue
                        
                    # Clean the word
                    word = re.sub(r'[^\w\s]', '', word.lower())
                    if word and len(word) > 1:
                        clean_keywords.append((word, weight))
                
                # Combine with unigram keywords for better context
                combined_features = []
                for word, weight in clean_keywords:
                    combined_features.append((word, weight))
                
                # Add top ngrams
                for feature, count in zip(features, counts):
                    if count > 1:  # Only include ngrams that appear more than once
                        combined_features.append((feature, float(count)))
                
                # Sort by weight
                combined_features.sort(key=lambda x: x[1], reverse=True)
                
                # Filter out ngrams containing non-english or non-meaningful terms
                filtered_features = [ (feat, wt) for feat, wt in combined_features 
                                  if not any(term in feat for term in ["http", "www", "com", "org", "gov", "edu"]) ]
                if filtered_features:
                    topic_label = filtered_features[0][0].title()
                else:
                    topic_label = processed_keywords[0].title() if processed_keywords else "Unlabeled Topic"
                subtopic_label = "General"

            except Exception as e:
                logger.error(f"Error parsing topic label: {e}")
                topic_label = processed_keywords[0].title() if processed_keywords else "Unlabeled Topic"
                subtopic_label = ' '.join(processed_keywords[1:3]).title() if len(processed_keywords) > 1 else "General"
            
            # Clean up labels
            topic_label = topic_label.replace('"', '').replace("'", "").strip()
            subtopic_label = subtopic_label.replace('"', '').replace("'", "").strip()
            
            # Truncate if too long
            if len(topic_label) > 30:
                topic_label = topic_label[:30]
            if len(subtopic_label) > 50:
                subtopic_label = subtopic_label[:50]
                
            # Cache result for future use
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                cache = {}
                if os.path.exists(cache_file):
                    with open(cache_file, 'r') as f:
                        cache = json.load(f)
                cache[cache_key] = [topic_label, subtopic_label]  # Store as list to avoid parsing issues
                with open(cache_file, 'w') as f:
                    json.dump(cache, f)
            except Exception as e:
                logger.warning(f"Error caching result: {e}")
                
            return topic_label, subtopic_label
            
        except Exception as e:
            logger.error(f"Error generating topic label: {e}")
            logger.error(traceback.format_exc())
            # Ensure we always return two values even in error cases
            if keywords and len(keywords) > 0:
                try:
                    if isinstance(keywords[0], tuple) and len(keywords[0]) >= 1:
                        topic = str(keywords[0][0]).replace('_', ' ').title()
                    else:
                        topic = str(keywords[0]).replace('_', ' ').title()
                    return topic, "General"
                except:
                    pass
            return "Unlabeled Topic", "General"

    def save_topic_keywords(self, topic_model: BERTopic) -> pd.DataFrame:
        """Generate and save topic keywords with labels."""
        try:
            # Create temp directory if it doesn't exist
            temp_dir = os.path.join(gl.output_folder, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            checkpoint_file = os.path.join(temp_dir, 'topic_keywords_checkpoint.pkl')
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'rb') as f:
                    return pickle.load(f)
                
            # Get basic topic info
            topic_info = topic_model.get_topic_info()
            
            # Add representative documents
            rep_docs = topic_model.representative_docs_ if hasattr(topic_model, "representative_docs_") else {}
            topic_info['Representative_Docs'] = topic_info['Topic'].map(lambda x: rep_docs.get(x, []))
            
            # Generate labels (split into topic and subtopic)
            main_topics = []
            subtopics = []
            for _, row in tqdm(topic_info.iterrows(), desc="Generating topic labels"):
                if row['Topic'] == -1:
                    main_topics.append("No Topic")
                    subtopics.append("")
                    continue
                keywords = [word for word, _ in topic_model.get_topics()[row['Topic']]]
                docs = row['Representative_Docs']
                topic_label, subtopic_label = self.generate_topic_label(keywords, docs)
                main_topics.append(topic_label)
                subtopics.append(subtopic_label)
            
            topic_info['Topic_Label'] = main_topics
            topic_info['Subtopic_Label'] = subtopics

            # Log the number of topics with labels
            logger.info(f"Generated labels for {len(main_topics)} topics")

            # Check if Topic_Label column is populated
            if 'Topic_Label' not in topic_info.columns or topic_info['Topic_Label'].isnull().all():
                logger.error("Topic_Label column is missing or not populated")
                raise ValueError("Topic_Label column is missing or not populated")

            output_path = os.path.join(
                gl.output_folder, 
                f"topic_keywords_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{self.n_topics}_{gl.YEAR_START}_{gl.YEAR_END}.csv"
            )
            topic_info.to_csv(output_path, index=False)
            logger.info(f"Saved topic keywords with labels to {output_path}")
            return topic_info
        except Exception as e:
            logger.error(f"Error in save_topic_keywords: {e}")
            raise

    

    def _create_custom_labels(self, topic_info: pd.DataFrame) -> dict:
        """Create custom labels for topics based on topic info.
        
        Args:
            topic_info: DataFrame with topic information
            
        Returns:
            Dictionary mapping topic IDs to custom labels
        """
        custom_labels = {}
        
        # Log the start of label creation process
        self.logger.info(f"Creating custom labels for {len(topic_info)} topics using OpenAI API")
        self.logger.info(f"Topic info columns: {list(topic_info.columns)}")
        
        # Check if Topic column exists
        if 'Topic' not in topic_info.columns:
            self.logger.error("Topic column missing from topic_info DataFrame")
            return custom_labels
        
        # Track success/failure for diagnostics
        success_count = 0
        failure_count = 0
        
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            
            # Skip the -1 topic (outliers)
            if topic_id == -1:
                custom_labels[-1] = "Outliers"
                continue
                
            try:
                # Extract keywords from the Name column
                # The Name column often contains topic keywords in format 'word_1 + word_2 + word_3'
                keywords = []
                if 'Name' in topic_info.columns and not pd.isna(row['Name']):
                    # Extract words from the Name column
                    name_text = str(row['Name'])
                    # Remove topic prefix if present
                    if name_text.startswith('Topic'):
                        name_text = name_text.split(':', 1)[-1].strip()
                    
                    # Split on '+' or spaces
                    if '+' in name_text:
                        keywords = [k.strip() for k in name_text.split('+')]
                    else:
                        # Just use words from the name
                        keywords = [w for w in name_text.split() if len(w) > 2]
                
                # If no keywords found, use placeholder
                if not keywords:
                    keywords = [f"topic_{topic_id}_keyword"]
                
                    # Get representative documents if available
                    docs = []
                if 'Representative_Docs' in row and not pd.isna(row['Representative_Docs']):
                    docs_str = row['Representative_Docs']
                    if isinstance(docs_str, str):
                        # Clean up the string representation of the list
                        docs_str = docs_str.strip('[]')
                        # Split by commas that aren't within quotes
                        import re
                        docs = re.findall(r'"([^"]*)"', docs_str)
                        if not docs:
                            # Try another approach if regex didn't work
                            docs = [d.strip().strip('"\'') for d in docs_str.split(',')]

                # If rep_docs is already a list or numpy array, convert each item to a string.
                elif isinstance(docs_str, (list, np.ndarray)):
                    docs = [str(d) for d in docs_str if d is not None]
                else:
                    # Otherwise, leave docs as empty.
                    docs = []
                                
                # Generate a label based on keywords and documents
                label = self.generate_topic_label(keywords, docs)
                
                # Store the label
                custom_labels[topic_id] = label
                success_count += 1
                
            except Exception as e:
                # Log error but continue processing other topics
                self.logger.error(f"Error creating label for topic {topic_id}: {e}")
                self.logger.error(f"Row data: {dict(row)}")  # Convert row to dict for cleaner logging
                custom_labels[topic_id] = f"Topic {topic_id}"
                failure_count += 1
        
        # Log completion statistics
        self.logger.info(f"Generated labels for {len(custom_labels)} topics (success: {success_count}, failure: {failure_count})")
        return custom_labels

    def update_topic_labels(self, topic_info: pd.DataFrame, topic_model: BERTopic) -> BERTopic:
        """
        Update topic model with custom labels.
        
        Args:
            topic_info: DataFrame with topic information
            topic_model: BERTopic model to update
            
        Returns:
            Updated BERTopic model
        """
        try:
            self.logger.info(f"Updating topic labels using DataFrame with columns: {list(topic_info.columns)}")
            
            # Create custom labels
            custom_labels = self._create_custom_labels(topic_info)
            
            # Log custom labels for debugging
            self.logger.info("Custom topic labels:")
            for topic_id, label in sorted(custom_labels.items()):
                if topic_id >= 0:  # Skip outlier topic (-1)
                    self.logger.info(f"  Topic {topic_id}: {label}")
            
            # Update the model with custom labels
            topic_model.set_topic_labels(custom_labels)
            self.logger.info(f"Updated topic model with {len(custom_labels)} custom labels")
            
            # Verify the labels were applied correctly
            if hasattr(topic_model, 'custom_labels_'):
                self.logger.info(f"Topic model now has {len(topic_model.custom_labels_)} custom labels")
            else:
                self.logger.warning("Topic model does not have custom_labels_ attribute after update")
                
                # Check if this is an older version of BERTopic that uses a different attribute
                if hasattr(topic_model, 'topic_labels_'):
                    self.logger.info(f"Using older BERTopic version with topic_labels_ attribute: {len(topic_model.topic_labels_)} labels")
            
            # Also update the n_topics attribute used for file naming
            if hasattr(topic_model, 'topics_'):
                unique_topics = set(topic_model.topics_)
                self.n_topics = len(unique_topics) - (1 if -1 in unique_topics else 0)
                self.logger.info(f"Updated n_topics to {self.n_topics}")
            
            return topic_model
        except Exception as e:
            self.logger.error(f"Error updating topic labels: {e}")
            self.logger.error(traceback.format_exc())
            return topic_model

    def _debug_topic_representatives(self, topic_representatives):
        """Debug function to check the structure of topic representatives."""
        try:
            logger.info(f"Debugging topic representatives structure")
            logger.info(f"Total topics: {len(topic_representatives)}")
            
            # Check if we have any topics
            if not topic_representatives:
                logger.warning("No topics found in topic_representatives")
                return
            
            # Check a sample of topics
            sample_size = min(5, len(topic_representatives))
            sample_topics = list(topic_representatives.items())[:sample_size]
            
            for topic_key, topic_data in sample_topics:
                logger.info(f"Topic {topic_key}:")
                logger.info(f"  - Number of docs: {len(topic_data.get('docs', []))}")
                logger.info(f"  - Number of embeddings: {len(topic_data.get('embeddings', []))}")
                
                # Check if docs and embeddings match
                if len(topic_data.get('docs', [])) != len(topic_data.get('embeddings', [])):
                    logger.warning(f"  - Mismatch: {len(topic_data.get('docs', []))} docs vs {len(topic_data.get('embeddings', []))} embeddings")
                
                # Check a sample document
                if topic_data.get('docs'):
                    sample_doc = topic_data['docs'][0]
                    logger.info(f"  - Sample doc: {sample_doc[:100]}...")  # First 100 chars
                
                # Check a sample embedding
                if topic_data.get('embeddings'):
                    sample_embedding = topic_data['embeddings'][0]
                    if isinstance(sample_embedding, list):
                        logger.info(f"  - Sample embedding: {len(sample_embedding)} dimensions, first 5: {sample_embedding[:5]}")
                    else:
                        logger.warning(f"  - Sample embedding is not a list: {type(sample_embedding)}")
            
        except Exception as e:
            logger.error(f"Error debugging topic representatives: {e}")
            logger.error(traceback.format_exc())

    def _create_topic_model(self, umap_model=None, hdbscan_model=None):
        """Create a BERTopic model with the specified components.
        
        Args:
            umap_model: UMAP model for dimensionality reduction
            hdbscan_model: HDBSCAN model for clustering
            
        Returns:
            Configured BERTopic model
        """
        try:
            # Log which embedding model we're using
            logger.info(f"Creating topic model with embedding model: {self.embedding_model}")
            
            # We don't need to create a SentenceTransformer instance here
            # since we're using pre-computed embeddings
            embedding_model = None
            
            # Create vectorizer model
            vectorizer_model = CountVectorizer(
                stop_words="english",
                min_df=gl.MIN_DF[0],
                max_df=gl.MAX_DF[0]
            )
            
            # Create default UMAP model if not provided
            if umap_model is None:
                umap_model = UMAP(
                    n_neighbors=gl.N_NEIGHBORS[0],
                    n_components=gl.N_COMPONENTS[0],
                    min_dist=gl.MIN_DIST[0],
                    metric='cosine',
                    random_state=42
                )
            
            # Create default HDBSCAN model if not provided
            if hdbscan_model is None:
                hdbscan_model = HDBSCAN(
                    min_samples=gl.MIN_SAMPLES[0],
                    min_cluster_size=gl.MIN_CLUSTER_SIZE[0],
                    metric='euclidean',
                    cluster_selection_method='eom',
                    prediction_data=True
                )
            
            # Create BERTopic model
            topic_model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                nr_topics=self.n_topics,
                top_n_words=gl.TOP_N_WORDS[0],
                calculate_probabilities=False,
                verbose=True
            )
            
            return topic_model
        except Exception as e:
            logger.error(f"Error creating topic model: {e}")
            logger.error(traceback.format_exc())
            raise

    def _save_umap_embeddings(self, topic_model, embeddings):
        """Save UMAP embeddings for future use to avoid recomputing them."""
        try:
            if hasattr(topic_model, 'umap_model') and topic_model.umap_model is not None:
                # Create a filename based on UMAP parameters
                umap_model = topic_model.umap_model
                params_str = f"n{umap_model.n_neighbors}_c{umap_model.n_components}_d{umap_model.min_dist:.2f}"
                umap_file = os.path.join(gl.output_folder, f'umap_embeddings_{params_str}.npy')
                
                # Check if file already exists
                if not os.path.exists(umap_file):
                    logger.info(f"Saving UMAP embeddings with parameters {params_str} for future use")
                    # Transform embeddings using UMAP
                    umap_embeddings = topic_model.umap_model.transform(embeddings)
                    # Save to file
                    np.save(umap_file, umap_embeddings)
                    logger.info(f"UMAP embeddings saved to {umap_file}")
                    return umap_embeddings
                else:
                    logger.info(f"UMAP embeddings file already exists at {umap_file}")
        except Exception as e:
            logger.warning(f"Error saving UMAP embeddings: {e}")
        return None
        
    def _load_umap_embeddings(self, umap_model):
        """Load pre-computed UMAP embeddings if available."""
        try:
            # Create a filename based on UMAP parameters
            params_str = f"n{umap_model.n_neighbors}_c{umap_model.n_components}_d{umap_model.min_dist:.2f}"
            umap_file = os.path.join(gl.output_folder, f'umap_embeddings_{params_str}.npy')
            
            if os.path.exists(umap_file):
                logger.info(f"Loading pre-computed UMAP embeddings from {umap_file}")
                umap_embeddings = np.load(umap_file)
                logger.info(f"Loaded UMAP embeddings with shape {umap_embeddings.shape}")
                return umap_embeddings
        except Exception as e:
            logger.warning(f"Error loading UMAP embeddings: {e}")
        return None

    def _store_representatives_fast(self, chunk_model, chunk_docs, chunk_embeddings, topic_representatives, topic_to_docs):
        """Optimized version of store_representatives that uses pre-computed topic assignments.
        
        Args:
            chunk_model: The BERTopic model
            chunk_docs: List of documents in the chunk
            chunk_embeddings: Embeddings of documents in the chunk
            topic_representatives: Dictionary to store representative documents
            topic_to_docs: Dictionary mapping topic IDs to document indices
        """
        try:
            # Count how many documents we're storing
            total_docs_stored = 0
            skipped_topics = 0
            
            # Get topic assignments
            topics = chunk_model.topics_
            
            # Process each topic (except outlier topic -1)
            for topic_id in set(topics):
                if topic_id == -1:  # Skip outlier topic
                    continue
                
                # Get all document indices for this topic
                doc_indices = topic_to_docs.get(topic_id, [])
                
                # Skip if we have no documents for this topic
                if not doc_indices:
                    continue
                
                # Skip topics with fewer than MIN_DOCS_PER_TOPIC documents (50)
                if len(doc_indices) < gl.MIN_DOCS_PER_TOPIC:
                    skipped_topics += 1
                    continue
                
                # Use all docs if less than MAX_DOCS_PER_TOPIC (300), otherwise limit to 300
                max_docs_to_store = min(gl.MAX_DOCS_PER_TOPIC, len(doc_indices))
                
                # Calculate how many documents we'll store for this topic
                if len(doc_indices) <= gl.MAX_DOCS_PER_TOPIC:
                    # Using all documents (between 50 and 300)
                    self.logger.debug(f"Topic {topic_id}: Using all {len(doc_indices)} documents")
                else:
                    # Limiting to MAX_DOCS_PER_TOPIC (300)
                    self.logger.debug(f"Topic {topic_id}: Limiting from {len(doc_indices)} to {max_docs_to_store} documents")
                
                doc_indices = doc_indices[:max_docs_to_store]
                
                # Collect documents and embeddings
                valid_docs = []
                valid_embeddings = []
                
                for idx in doc_indices:
                    if idx < len(chunk_docs):
                        # Add document and embedding
                        valid_docs.append(chunk_docs[idx])
                        # Store the embedding as a list for JSON serialization
                        valid_embeddings.append(chunk_embeddings[idx].tolist())
                
                # Verify we still have enough valid documents
                if len(valid_docs) < gl.MIN_DOCS_PER_TOPIC:
                    skipped_topics += 1
                    continue
                
                # Store the documents and embeddings
                if valid_docs and valid_embeddings:
                    topic_key = f"topic_{len(topic_representatives)}"
                    topic_representatives[topic_key] = {
                        'docs': valid_docs,
                        'embeddings': valid_embeddings
                    }
                    total_docs_stored += len(valid_docs)
            
            logger.info(f"Stored representatives for {len(topic_representatives)} topics with a total of {total_docs_stored} documents")
            if skipped_topics > 0:
                logger.info(f"Skipped {skipped_topics} topics with fewer than {gl.MIN_DOCS_PER_TOPIC} documents")
            
        except Exception as e:
            logger.error(f"Error storing representatives: {e}")
            logger.error(traceback.format_exc())
            raise

    def _process_chunks_parallel(self, docs: list, embeddings: np.ndarray, n_workers: int = 4) -> dict:
        """Process documents in chunks using parallel processing.
        
        Args:
            docs: List of documents to process
            embeddings: Document embeddings
            n_workers: Number of worker processes to use
            
        Returns:
            Dictionary of topic representatives
        """
        try:
            from multiprocessing import Pool, Manager
            import numpy as np
            
            # Initialize shared dictionary for topic representatives
            manager = Manager()
            topic_representatives = manager.dict()
            
            # Calculate optimal chunk size based on GPU memory
            chunk_size = self._calculate_optimal_batch_size()
            logger.info(f"Processing documents in chunks of size {chunk_size} using {n_workers} workers")
            
            # Calculate number of chunks
            n_chunks = (len(docs) + chunk_size - 1) // chunk_size
            
            # Create chunk parameters
            chunk_params = []
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(docs))
                
                # Skip if no documents in this chunk
                if start_idx >= len(docs):
                    continue
                
                # Get chunk of documents and embeddings
                chunk_docs = docs[start_idx:end_idx]
                chunk_embeddings = embeddings[start_idx:end_idx].copy()  # Create a copy for sharing
                
                # Add chunk parameters
                chunk_params.append((i, chunk_docs, chunk_embeddings, n_chunks))
            
            # Define worker function
            def process_chunk(params):
                try:
                    chunk_idx, chunk_docs, chunk_embeddings, total_chunks = params
                    
                    logger.info(f"Worker processing chunk {chunk_idx+1}/{total_chunks} with {len(chunk_docs)} documents")
                    
                    # Create models for this worker
                    n_neighbors = 5
                    n_components = 50
                    min_dist = 0.0
                    min_samples = 5
                    min_cluster_size = 30
                    
                    umap_model = UMAP(
                        n_neighbors=n_neighbors,
                        n_components=n_components,
                        min_dist=min_dist,
                        metric='cosine',
                        random_state=42,
                        verbose=False,
                        low_memory=True,
                        n_jobs=1  # Use 1 job per worker to avoid nested parallelism
                    )
                    
                    hdbscan_model = HDBSCAN(
                        min_samples=min_samples,
                        min_cluster_size=min_cluster_size,
                        metric='euclidean',
                        cluster_selection_method='eom',
                        prediction_data=True,
                        core_dist_n_jobs=1  # Use 1 job per worker
                    )
                    
                    # Create a BERTopic model instance for this worker
                    chunk_model = BERTopic(
                        umap_model=umap_model,
                        hdbscan_model=hdbscan_model,
                        embedding_model=None,
                        calculate_probabilities=False,
                        verbose=False
                    )
                    
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
                        
                        # Take up to 30 documents (increased from 20)
                        max_docs = max(gl.MAX_DOCS_PER_TOPIC, len(doc_indices))
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
                    
                    # Return the representatives for this chunk
                    return chunk_representatives
                    
                except Exception as e:
                    logger.error(f"Worker error processing chunk {chunk_idx+1}: {e}")
                    logger.error(traceback.format_exc())
                    return {}
            
            # Use a process pool to process chunks in parallel
            with Pool(processes=n_workers) as pool:
                results = pool.map(process_chunk, chunk_params)
            
            # Combine results from all workers
            combined_representatives = {}
            for chunk_result in results:
                combined_representatives.update(chunk_result)
            
            # Save final representatives
            representatives_path = os.path.join(self.output_dirs['models'], 'topic_representatives.json')
            with open(representatives_path, 'w', encoding='utf-8') as f:
                json.dump(combined_representatives, f, ensure_ascii=False, indent=2)
            
            # Log final statistics
            total_topics = len(combined_representatives)
            total_docs = sum(len(topic_data.get('docs', [])) for topic_data in combined_representatives.values())
            logger.info(f"Parallel processing completed with {total_topics} total topics and {total_docs} representative documents")
            
            return combined_representatives
            
        except Exception as e:
            logger.error(f"Error in parallel chunk processing: {e}")
            logger.error(traceback.format_exc())
            raise

    def _setup_folder_structure(self):
        """
        Setup the necessary folder structure for the topic modeler.
        Creates all the required directories for outputs, models, embeddings, etc.
        """
        # Define the directories we need
        self.output_dirs = {
            'temp': os.path.join(gl.output_folder, 'temp'),
            'models': os.path.join(gl.output_folder, 'models'),
            'embeddings': os.path.join(gl.output_folder, 'temp', 'embeddings'),
            'figures': os.path.join(gl.output_folder, 'figures')
        }
        
        # Create each directory if it doesn't exist
        for name, directory in self.output_dirs.items():
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
        
        # Set batch size based on device and model - moved from __init__
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_mem_gb = gpu_props.total_memory / (1024**3)
            
            self.logger.info(f"CUDA Device: {gpu_props.name}")
            self.logger.info(f"CUDA Memory: {gpu_mem_gb:.2f} GB")
            
            # Adjust batch size based on model size and GPU memory
            if 'large' in self.embedding_model.lower() or 'bge' in self.embedding_model.lower():
                # For larger models like BGE-large, gte-large, etc.
                if gpu_mem_gb > 35:  # A100 40GB or similar
                    self.base_batch_size = 384
                else:
                    self.base_batch_size = 128
            elif gpu_mem_gb > 35:  # For smaller models on high memory GPUs
                self.base_batch_size = gl.GPU_BATCH_SIZE
            else:
                self.base_batch_size = 256
        else:
            self.base_batch_size = 64
            self.logger.info("Using CPU")
        
        self.logger.info(f"Base batch size set to {self.base_batch_size}")
        self.logger.info("Folder structure setup complete")

    def _calculate_optimal_batch_size(self) -> int:
        """
        Calculate the optimal batch size based on available GPU memory.
        
        Returns:
            int: Optimal batch size for processing
        """
        try:
            # Default values if detection fails
            default_batch_size = 50000
            
            if self.device == 'cuda':
                # Check if we have GPU
                import torch
                if torch.cuda.is_available():
                    # Get GPU info
                    device_idx = 0  # Use first GPU by default
                    gpu_memory = torch.cuda.get_device_properties(device_idx).total_memory
                    
                    # Convert to GB
                    gpu_memory_gb = gpu_memory / (1024**3)
                    
                    self.logger.info(f"Detected GPU with {gpu_memory_gb:.2f} GB memory")
                    
                    # Calculate batch size based on GPU memory
                    # Very conservative estimate based on embedding size and overhead
                    # For 1024-dimensional embeddings, ~4KB per embedding
                    # Plus overhead for UMAP, HDBSCAN, etc.
                    
                    if gpu_memory_gb > 35:  # A100 40GB or similar
                        return 100000
                    elif gpu_memory_gb > 20:  # V100 32GB or similar
                        return 75000
                    elif gpu_memory_gb > 10:  # V100 16GB, RTX 3090, etc.
                        return 50000
                    elif gpu_memory_gb > 6:  # RTX 2080 Ti, etc.
                        return 30000
                    else:  # Smaller GPUs
                        return 15000
                else:
                    self.logger.warning("CUDA device requested but not available. Using CPU batch size.")
                    return default_batch_size
            else:
                # CPU-based sizing
                import psutil
                memory_gb = psutil.virtual_memory().total / (1024**3)
                
                self.logger.info(f"Running on CPU with {memory_gb:.2f} GB system memory")
                
                # CPU can handle larger batches but is slower
                if memory_gb > 200:  # High-memory server
                    return 100000
                elif memory_gb > 64:
                    return 75000
                elif memory_gb > 32:
                    return 50000
                elif memory_gb > 16:
                    return 30000
                else:
                    return 15000
        except Exception as e:
            self.logger.warning(f"Error detecting optimal batch size: {e}")
            self.logger.warning(f"Using default batch size of {default_batch_size}")
            return default_batch_size

    def _calculate_adaptive_parameters(self, docs: list, embeddings: np.ndarray) -> dict:
        """
        Calculate adaptive parameters for earnings call topic modeling.
        
        This method analyzes the input data and determines optimal parameters for:
        - Clustering (HDBSCAN)
        - Dimensionality reduction (UMAP)
        - Topic modeling (BERTopic)
        
        The parameters are adapted to the specific characteristics of earnings calls:
        - Financial terminology and domain-specific language
        - Quarterly reporting patterns
        - Company-specific vs. industry-wide topics
        - Forward-looking statements vs. historical reporting
        
        Args:
            docs: List of documents (earnings call transcripts)
            embeddings: Document embeddings
            
        Returns:
            Dictionary of adaptive parameters
        """
        self.logger.info("Calculating adaptive parameters for earnings call topic modeling...")
        
        # Starting with default parameters
        parameters = {
            # HDBSCAN parameters
            "min_cluster_size": gl.MIN_CLUSTER_SIZE[0] if hasattr(gl, 'MIN_CLUSTER_SIZE') and gl.MIN_CLUSTER_SIZE else 10,
            "min_samples": 5,
            "cluster_selection_epsilon": 0.2,
            
            # UMAP parameters
            "n_neighbors": gl.N_NEIGHBORS[0] if hasattr(gl, 'N_NEIGHBORS') and gl.N_NEIGHBORS else 15,
            "n_components": gl.N_COMPONENTS[0] if hasattr(gl, 'N_COMPONENTS') and gl.N_COMPONENTS else 5,
            "min_dist": 0.1,
            
            # Topic modeling parameters
            "n_topics": self.n_topics if hasattr(self, 'n_topics') else "auto",
            "top_n_words": 10,
            
            # Additional parameters for earnings calls
            "min_topic_size": 5,  # Minimum number of documents to form a topic
            "use_sentiment": True,  # Consider sentiment in earnings calls
            "temporal_weight": 0.5,  # Weight for temporal aspects (0-1)
        }
        
        # 1. Adapt based on corpus size
        n_docs = len(docs)
        self.logger.info(f"Adapting parameters for corpus size: {n_docs} documents")
        
        if n_docs < 100:
            # Small corpus: smaller clusters, fewer neighbors
            parameters["min_cluster_size"] = max(3, int(n_docs * 0.05))
            parameters["n_neighbors"] = max(5, int(n_docs * 0.1))
            parameters["min_topic_size"] = 2
        elif n_docs < 500:
            # Medium corpus
            parameters["min_cluster_size"] = max(5, int(n_docs * 0.03))
            parameters["n_neighbors"] = max(10, int(n_docs * 0.05))
            parameters["min_topic_size"] = 3
        elif n_docs < 2000:
            # Large corpus
            parameters["min_cluster_size"] = max(10, int(n_docs * 0.02))
            parameters["n_neighbors"] = max(15, int(n_docs * 0.03))
            parameters["min_topic_size"] = 5
        else:
            # Very large corpus
            parameters["min_cluster_size"] = max(20, int(n_docs * 0.01))
            parameters["n_neighbors"] = max(30, int(n_docs * 0.015))
            parameters["min_topic_size"] = 10
            
        # Adaptive min_samples based on min_cluster_size
        parameters["min_samples"] = max(3, parameters["min_cluster_size"] // 5)
        
        # 2. Adapt based on embedding dimensionality
        embedding_dim = embeddings.shape[1] if embeddings is not None and len(embeddings.shape) > 1 else 0
        self.logger.info(f"Adapting parameters for embedding dimensionality: {embedding_dim}")
        
        if embedding_dim > 0:
            # Adjust n_components based on embedding dimensionality
            # Higher dimensional embeddings may require more components
            parameters["n_components"] = min(max(5, embedding_dim // 100), 50)
        
        # 3. Analyze document length distribution for earnings calls
        doc_lengths = [len(doc.split()) for doc in docs]
        avg_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
        self.logger.info(f"Average document length: {avg_length:.1f} words")
        
        if avg_length > 3000:
            # Very detailed earnings calls with extensive Q&A
            # Need more fine-grained topics and cluster distinction
            parameters["min_dist"] = 0.05  # Tighter clustering in UMAP space
            parameters["cluster_selection_epsilon"] = 0.15  # More precise clusters
            
            # For long earnings calls, we likely need more topics to capture nuances
            if parameters["n_topics"] == "auto":
                parameters["n_topics"] = max(20, n_docs // 20)
        elif avg_length > 1500:
            # Standard earnings calls
            parameters["min_dist"] = 0.1
            parameters["cluster_selection_epsilon"] = 0.2
            if parameters["n_topics"] == "auto":
                parameters["n_topics"] = max(15, n_docs // 30)
        else:
            # Shorter earnings summaries
            parameters["min_dist"] = 0.2  # More spread out clusters
            parameters["cluster_selection_epsilon"] = 0.3  # More inclusive clusters
            if parameters["n_topics"] == "auto":
                parameters["n_topics"] = max(10, n_docs // 40)
        
        # 4. Detect financial quarter patterns
        quarter_patterns = self._detect_quarterly_patterns(docs)
        if quarter_patterns:
            self.logger.info("Detected quarterly reporting patterns")
            # If we detect strong quarterly patterns, adjust for temporal coherence
            parameters["temporal_weight"] = 0.7  # Increase temporal importance
        
        # 5. Look for industry-specific clusters
        industry_clusters = self._detect_industry_clusters(docs)
        if industry_clusters:
            self.logger.info(f"Detected {len(industry_clusters)} potential industry clusters")
            # If we detect strong industry clustering, reduce min_cluster_size
            # to allow for industry-specific smaller topics
            parameters["min_cluster_size"] = max(3, parameters["min_cluster_size"] - 2)
        
        # 6. Final adjustments and logging
        self.logger.info("Final adaptive parameters:")
        for key, value in parameters.items():
            self.logger.info(f"  - {key}: {value}")
        
        # Set the parameters for future use
        self.adaptive_parameters = parameters
        
        # Convert parameter settings to global options
        gl.MIN_CLUSTER_SIZE = [parameters["min_cluster_size"]]
        gl.N_NEIGHBORS = [parameters["n_neighbors"]]
        gl.N_COMPONENTS = [parameters["n_components"]]
        self.n_topics = parameters["n_topics"]
        
        return parameters
    
    def _detect_quarterly_patterns(self, docs: list) -> bool:
        """
        Detect quarterly reporting patterns in earnings call documents.
        
        Args:
            docs: List of documents (earnings call transcripts)
            
        Returns:
            Boolean indicating if quarterly patterns were detected
        """
        # Simple check for quarterly terms in the documents
        quarterly_terms = ["Q1", "Q2", "Q3", "Q4", "first quarter", "second quarter", 
                          "third quarter", "fourth quarter", "quarterly", "year-over-year",
                          "year over year", "YoY", "quarter-over-quarter", "QoQ"]
        
        # Sample a subset of documents for efficiency
        sample_size = min(100, len(docs))
        sample_docs = random.sample(docs, sample_size) if len(docs) > sample_size else docs
        
        # Count documents with quarterly terms
        docs_with_quarterly_terms = 0
        for doc in sample_docs:
            if any(term in doc for term in quarterly_terms):
                docs_with_quarterly_terms += 1
        
        # Calculate percentage of documents with quarterly terms
        quarterly_percentage = docs_with_quarterly_terms / len(sample_docs) if sample_docs else 0
        
        # Return True if a significant percentage of documents contain quarterly terms
        return quarterly_percentage > 0.3
    
    def _detect_industry_clusters(self, docs: list) -> list:
        """
        Detect potential industry-specific clusters in earnings call documents.
        
        Args:
            docs: List of documents (earnings call transcripts)
            
        Returns:
            List of potential industry clusters
        """
        # Industry-specific terms dictionary
        industry_terms = {
            "technology": ["software", "hardware", "tech", "digital", "cloud", "AI", "artificial intelligence"],
            "finance": ["banking", "investment", "financial", "asset", "portfolio", "loan", "deposit"],
            "healthcare": ["medical", "pharma", "healthcare", "patient", "clinical", "drug", "therapeutic"],
            "energy": ["oil", "gas", "energy", "renewable", "solar", "wind", "fossil", "petroleum"],
            "retail": ["retail", "e-commerce", "store", "consumer", "merchandise", "inventory", "sales"],
            "manufacturing": ["manufacturing", "factory", "production", "supply chain", "industrial"],
            "real_estate": ["property", "real estate", "lease", "rental", "mortgage", "construction"]
        }
        
        # Sample a subset of documents for efficiency
        sample_size = min(200, len(docs))
        sample_docs = random.sample(docs, sample_size) if len(docs) > sample_size else docs
        
        # Count industry term occurrences
        industry_counts = {industry: 0 for industry in industry_terms}
        for doc in sample_docs:
            doc_lower = doc.lower()
            for industry, terms in industry_terms.items():
                if any(term.lower() in doc_lower for term in terms):
                    industry_counts[industry] += 1
        
        # Calculate percentages
        industry_percentages = {industry: count / len(sample_docs) for industry, count in industry_counts.items()}
        
        # Identify significant industries (present in at least 20% of documents)
        significant_industries = [industry for industry, percentage in industry_percentages.items() 
                               if percentage > 0.2]
        
        return significant_industries