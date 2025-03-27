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
from eCallsAgent.core.chunking_utils import _cpu_topic_model, _gpu_topic_model
import json
import pickle
import gc
import pandas as pd
import csv
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
from datetime import datetime
import plotly.io as pio
import os
from multiprocessing import Pool, cpu_count
import numpy as np
from eCallsAgent.core.visualization import TopicVis
from eCallsAgent.core.chunking_utils import process_chunk_worker  # Add this import
from eCallsAgent.utils.cuda_setup import setup_cuda, check_cuml_availability 
from eCallsAgent.core.embedding_generator import EmbeddingGenerator
from eCallsAgent.config import global_options as gl # global settings
# Configure Numba and CUDA logger levels
ENABLE_NUMBA_CUDA = os.environ.get('ENABLE_NUMBA_CUDA', 'True').lower() in ('true', '1', 't')
numba_logger = logging.getLogger('numba.cuda.cudadrv.driver')
numba_logger.setLevel(logging.CRITICAL)  # Suppress Numba CUDA driver errors
ptx_logger = logging.getLogger('ptxcompiler.patch')
ptx_logger.setLevel(logging.CRITICAL)  # Suppress ptxcompiler patch errors

# Import the cuda_setup utility

# Run CUDA setup before imports
CUDA_READY, CUDA_DEVICE, CUDA_MEMORY = setup_cuda()
CUML_AVAILABLE = check_cuml_availability() if CUDA_READY else False

# Conditionally import cuML if available
if CUML_AVAILABLE and CUDA_READY:
    try:
        from cuml import UMAP as cumlUMAP
        from cuml.cluster import HDBSCAN as cumlHDBSCAN
        logger = logging.getLogger("eCallsAgent.core.topic_modeler")
        logger.info("Successfully imported GPU-accelerated models (cuML)")
    except ImportError:
        CUML_AVAILABLE = False
        logger = logging.getLogger("eCallsAgent.core.topic_modeler")
        logger.warning("GPU-accelerated models (cuML) not available despite CUDA being ready")
    except Exception as e:
        CUML_AVAILABLE = False
        logger = logging.getLogger("eCallsAgent.core.topic_modeler")
        logger.warning(f"Error importing GPU-accelerated models: {e}")

# Add this near the top of the file, after other imports
logger = logging.getLogger("eCallsAgent.core.topic_modeler")

try:
    import importlib.util
    import subprocess
    
    # Log system information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"CUDA available via PyTorch: {torch.cuda.is_available()}")
    logger.info(f"CUDA initialized successfully: {CUDA_READY}")
    logger.info(f"cuML available: {CUML_AVAILABLE}")
    
    if CUDA_READY:
        logger.info(f"Using CUDA device: {CUDA_DEVICE} with {CUDA_MEMORY:.2f} GB memory")
except Exception as e:
    logger.error(f"Error in setup section: {e}")
    # Re-raise or handle as appropriate
    raise

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
        self.logger.info(f"Using embedding model: {self.embedding_model}")
        
        # Add default value for n_topics
        self.n_topics = gl.NR_TOPICS[0]
        
        # Initialize model evaluator
        self.model_evaluator = ModelEvaluator()

        # Initialize topic model
        self.topic_model = None
        

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
            self.logger.info(f"No chunk size provided, using calculated optimal size: {chunk_size}")
        
        # Use optimal parameters from grid search results
        n_neighbors = 5       # Grid search optimal value
        n_components = 50     # Grid search optimal value
        min_dist = 0.0        # Grid search optimal value
        min_samples = 5       # Grid search optimal value
        min_cluster_size = 30 # Grid search optimal value
        
        self.logger.info(f"Using grid search optimal parameters: n_neighbors={n_neighbors}, n_components={n_components}, min_dist={min_dist}")
        self.logger.info(f"HDBSCAN parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        self.logger.info(f"Creating chunk model with chunk size: {chunk_size}")
        
        return BERTopic(
            umap_model=UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                metric='cosine',
                random_state=42,
                verbose=False,
                low_memory=False,
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
        self.logger.info(f"Using GPU-accelerated document transformation for {len(docs)} documents")
        
        try:
            # Determine available GPU resources
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                self.logger.warning("No GPUs available, falling back to CPU")
                # Attempt to transform with CPU
                topics, _ = self.topic_model.transform(docs, embeddings)
                return topics
            
            gpus_per_node = min(4, num_gpus)  # Use at most 4 GPUs per node
            self.logger.info(f"Using {gpus_per_node} GPUs for document transformation")
            
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
                self.logger.info(f"Processing documents {start_idx}-{end_idx} on GPU {gpu_id}")
                
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
                        self.logger.error(f"Error processing chunk {chunk_start}-{chunk_end} on GPU {gpu_id}: {str(e)}")
                        # Assign noise label (-1) to failed chunks
                        gpu_results.extend([-1] * (chunk_end - chunk_start))
                
                # Collect results from this GPU
                all_topics.extend(gpu_results)
                
                # Log progress
                self.logger.info(f"Completed processing on GPU {gpu_id}, processed {len(gpu_results)} documents")
            
            # Verify results
            if len(all_topics) == 0:
                self.logger.warning("No topics were generated. Returning default assignments.")
                return [-1] * len(docs)
                
            if len(all_topics) != len(docs):
                self.logger.warning(f"Mismatch in results: got {len(all_topics)} labels for {len(docs)} documents")
                # Pad with noise labels if necessary
                if len(all_topics) < len(docs):
                    all_topics.extend([-1] * (len(docs) - len(all_topics)))
                else:
                    # Trim if somehow we got more labels than documents
                    all_topics = all_topics[:len(docs)]
            
            self.logger.info(f"Completed distributed transformation of {len(all_topics)} documents")
            return all_topics
            
        except Exception as e:
            self.logger.error(f"Error in distributed document transformation: {e}")
            self.logger.error(traceback.format_exc())
            # Return noise labels for all documents in case of failure
            return [-1] * len(docs)

    def train_topic_model(self, docs: list, embeddings: list, use_parallel=True) -> BERTopic:
        """
        Train a topic model on the given documents and embeddings.
        
        Args:
            docs: List of documents to model
            embeddings: List of embeddings for the documents
            use_parallel: Whether to use parallel processing for chunk processing
            
        Returns:
            BERTopic: Trained topic model
        """
        start_time = time.time()
        self.logger.info("Starting topic modeling...")
        
        try:
            # Convert embeddings to numpy array if they're not already
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            # Log the start of training
            start_time = time.time()
            self.logger.info(f"Starting topic modeling process with {len(docs)} documents")
            
            # Calculate adaptive parameters for earnings calls
            self.logger.info("**************Phase 0: Calculating adaptive parameters**************")
            self.adaptive_parameters = self._calculate_adaptive_parameters(docs, embeddings)
            self.logger.info(f"Using adaptive min_cluster_size: {self.adaptive_parameters['min_cluster_size']}, " + 
                            f"n_neighbors: {self.adaptive_parameters['n_neighbors']}, " +
                            f"n_components: {self.adaptive_parameters['n_components']}")
            
            # Preprocessing
            self.logger.info("**************Phase 1: Preprocessing data**************")
            preprocessed_docs, preprocessed_embeddings = self._preprocess_data(docs, embeddings)
            
            # Process the data in chunks to avoid memory issues
            self.logger.info("**************Phase 2: Processing data in chunks**************")    
            self.logger.info("Using parallel processing for chunk processing")
            topic_representatives = self._process_chunks_CPU_parallel(preprocessed_docs, preprocessed_embeddings, _cpu=True)
    
            # Distill the topics to get representative documents
            self.logger.info("**************Phase 3: Distilling topics to get representative documents**************")
            all_rep_docs, all_rep_embeddings, _ = self._distill_topics(topic_representatives, preprocessed_docs, preprocessed_embeddings)
            
            # Train the final model
            self.logger.info("**************Phase 4: Training final topic model on representative documents**************")
            final_model = self._train_final_model(all_rep_docs, all_rep_embeddings)

            # Clean up temporary files
            self._cleanup_temp_files()
            
            # Log completion
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.logger.info(f"**************Topic modeling completed in {elapsed_time:.2f} seconds**************")
            
            return final_model
            
        except Exception as e:
            self.logger.error(f"Error training topic model: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
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


    def _distill_topics(self, topic_representatives: dict, docs: list, embeddings: np.ndarray) -> tuple:
        """Distill topics from chunks to get representative documents."""
        try:
            # Determine the maximum number of representative documents
            max_representatives = self._calculate_max_representatives(len(docs))
            
            # Increase minimum and maximum docs per topic to ensure enough representation
            min_docs_per_topic = gl.MIN_DOCS_PER_TOPIC  # e.g., ensure at least 10 (or 30) docs per topic
            max_docs_per_topic = gl.MAX_DOCS_PER_TOPIC  # e.g., ensure at most 300 docs per topic
            
            # Set target number of topics
            target_topics = gl.NR_TOPICS[0]  # e.g., target to 300 topics
            
            # Log parameters for transparency
            self.logger.info(f"Distilling topics with parameters:")
            self.logger.info(f"  - Max representatives: {max_representatives}")
            self.logger.info(f"  - Min docs per topic: {min_docs_per_topic}")
            self.logger.info(f"  - Max docs per topic: {max_docs_per_topic}")
            self.logger.info(f"  - Target topics: {target_topics}")
            
            # Get the sorted list of topics
            sorted_topics = sorted(topic_representatives.keys())
            self.logger.info(f"Sorted {len(sorted_topics)} topics")
            
            # Adaptive target topics based on corpus size
            if len(docs) > 500000:
                target_topics = min(3000, len(sorted_topics))
                self.logger.info(f"Very large corpus: using up to {target_topics} topics")
            elif len(docs) > 100000:
                target_topics = min(2000, len(sorted_topics))
                self.logger.info(f"Large corpus: using up to {target_topics} topics")
            elif len(docs) > 50000:
                target_topics = min(1500, len(sorted_topics))
                self.logger.info(f"Medium corpus: using up to {target_topics} topics")
            else:
                target_topics = min(1000, len(sorted_topics))
                self.logger.info(f"Smaller corpus: using up to {target_topics} topics")
            
            # Calculate topic sizes
            topic_sizes = [len(topic_representatives[t].get('docs', [])) for t in sorted_topics[:target_topics]]
            
            if topic_sizes:
                # Calculate quartiles for topic sizes
                q1 = np.percentile(topic_sizes, 25)
                q2 = np.percentile(topic_sizes, 50)
                q3 = np.percentile(topic_sizes, 75)
                self.logger.info(f"Topic size quartiles: Q1={q1:.1f}, Q2={q2:.1f}, Q3={q3:.1f}")
                
                # Adjusted max docs per topic to lower the total sample size
                max_docs_ranges = {
                    "largest": min(1000, max(500, len(docs) // 1000)),
                    "large": min(500, max(200, len(docs) // 1500)),
                    "medium": min(200, max(100, len(docs) // 2000)),
                    "small": min(100, max(50, len(docs) // 2500))
                }
                self.logger.info(f"Using adaptive max docs per topic: {max_docs_ranges}")
                
                def get_max_docs(topic_size):
                    if topic_size >= q3:
                        return max_docs_ranges["largest"]
                    elif topic_size >= q2:
                        return max_docs_ranges["large"]
                    elif topic_size >= q1:
                        return max_docs_ranges["medium"]
                    else:
                        return max_docs_ranges["small"]
                
                distill_rep_docs = []
                distill_rep_embeddings = []
                topic_counts = {}
                
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
                    
                    distill_rep_docs.extend(topic_rep_docs)
                    distill_rep_embeddings.extend(topic_rep_embeddings)
                    topic_counts[topic] = topic_count
                    
                    self.logger.info(f"Topic {topic} has {topic_count} docs (max allowed: {max_docs}, original size: {topic_size})")
                
                if topic_counts:
                    avg_docs = sum(topic_counts.values()) / len(topic_counts)
                    min_docs = min(topic_counts.values())
                    max_docs_val = max(topic_counts.values())
                    self.logger.info(f"Topics processed: {len(topic_counts)}, Avg docs/topic: {avg_docs:.1f}, Min: {min_docs}, Max: {max_docs_val}")
                
                # --- Global Sampling: Enforce a maximum overall representative set ---
                MAX_GLOBAL_REPRESENTATIVES = gl.MAX_ADAPTIVE_REPRESENTATIVES  # Adjust this cap as needed
                if len(distill_rep_docs) > MAX_GLOBAL_REPRESENTATIVES:
                    self.logger.info(f"Total representative docs ({len(distill_rep_docs)}) exceeds global cap ({MAX_GLOBAL_REPRESENTATIVES}). Sampling down.")
                    indices = random.sample(range(len(distill_rep_docs)), MAX_GLOBAL_REPRESENTATIVES)
                    distill_rep_docs = [distill_rep_docs[i] for i in indices]
                    distill_rep_embeddings = [distill_rep_embeddings[i] for i in indices]
                
                return distill_rep_docs, distill_rep_embeddings, topic_counts
            else:
                # Fallback if topic_sizes is empty
                max_docs_per_topic = min(300, max(100, len(docs) // 800))
                self.logger.info(f"Using fixed max_docs_per_topic={max_docs_per_topic} based on corpus size")
                distill_rep_docs, distill_rep_embeddings, topic_counts = self._collect_representatives(
                    sorted_topics[:target_topics],
                    max_representatives,
                    gl.MIN_DOCS_PER_TOPIC,
                    max_docs_per_topic
                )
                return distill_rep_docs, distill_rep_embeddings, topic_counts
        except Exception as e:
            self.logger.error(f"Error in distill_topics: {e}")
            self.logger.error(traceback.format_exc())
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
            self.logger.warning(f"Topic {topic} has mismatched docs ({len(docs)}) and embeddings ({len(embeddings)})")
            # Take the minimum of the two to avoid errors
            min_length = min(len(docs), len(embeddings))
            docs = docs[:min_length]
            embeddings = embeddings[:min_length]
        
        # Determine how many documents to collect (between min and max)
        num_docs = min(max_docs, len(docs))
        
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

    def _train_final_model(self, all_rep_docs: list, all_rep_embeddings: list) -> BERTopic:
        """Train the final topic model on representative documents.
        Use GPU acceleration model for final model training"""
        self.logger.info(f"Training final model on {len(all_rep_docs)} representative documents")
        try:
            # Convert list of embeddings to a single numpy array
            if not isinstance(all_rep_embeddings, np.ndarray):
                self.logger.info(f"Converting embeddings from list to numpy array")
                all_rep_embeddings = np.array(all_rep_embeddings, dtype=np.float32)
                self.logger.info(f"Embeddings converted, shape: {all_rep_embeddings.shape}")

            # Get adaptive parameters
            self.adaptive_parameters = self._calculate_adaptive_parameters(all_rep_docs, all_rep_embeddings)
            
            # Use adaptive parameters if available
            if hasattr(self, 'adaptive_parameters'):
                n_neighbors = self.adaptive_parameters["n_neighbors"]
                n_components = self.adaptive_parameters["n_components"]
                min_dist = self.adaptive_parameters["min_dist"]
                min_cluster_size = self.adaptive_parameters["min_cluster_size"]
                min_samples = self.adaptive_parameters["min_samples"]
                cluster_selection_epsilon = self.adaptive_parameters["cluster_selection_epsilon"]
                
                self.logger.info(f"Using adaptive parameters for earnings call analysis:")
            else:
                # Fallback to default parameters
                n_neighbors = gl.N_NEIGHBORS[0] if hasattr(gl, 'N_NEIGHBORS') and gl.N_NEIGHBORS else 15
                n_components = gl.N_COMPONENTS[0] if hasattr(gl, 'N_COMPONENTS') and gl.N_COMPONENTS else 5
                min_dist = 0.1
                min_cluster_size = gl.MIN_CLUSTER_SIZE[0] if hasattr(gl, 'MIN_CLUSTER_SIZE') and gl.MIN_CLUSTER_SIZE else 10
                min_samples = max(3, min_cluster_size // 5)
                cluster_selection_epsilon = 0.2
                
                self.logger.info(f"Using default parameters (adaptive parameters not available):")
            
            self.logger.info(f"  - n_neighbors: {n_neighbors}")
            self.logger.info(f"  - n_components: {n_components}")
            self.logger.info(f"  - min_dist: {min_dist}")
            self.logger.info(f"  - min_cluster_size: {min_cluster_size}")
            self.logger.info(f"  - min_samples: {min_samples}")
            self.logger.info(f"  - cluster_selection_epsilon: {cluster_selection_epsilon}")
                        
            umap_model, hdbscan_model = _cpu_topic_model(n_neighbors, n_components, min_dist, min_samples, min_cluster_size, cluster_selection_epsilon)
            
            # Create and fit the model...
            # Create the BERTopic model with the optimized models
            topic_model = self._create_topic_model(umap_model, hdbscan_model)
            topic_model.verbose = True
            
            # Fit model with more verbosity
            self.logger.info("Fitting topic model on representative documents...")
            
            # Add validation to ensure documents are assigned to topics
            self.logger.info("Verifying topic assignments...")
            
            # Check if topic_model has been created successfully
            if not hasattr(self, 'topic_model') or self.topic_model is None:
                self.logger.warning("Topic model not created successfully. Creating default model...")
                self.topic_model = self._create_topic_model()
            
            # Fit the model
            self.logger.info("Fitting topic model to representative documents...")
            topics, probs = topic_model.fit_transform(all_rep_docs, all_rep_embeddings)
            
            # Store results and set the model
            self.rep_topics = topics
            self.rep_probs = probs
            
            # Now we can safely perform operations that require a fitted model
            self.logger.info("Model fitted successfully, now checking for empty topics...")
            
            # NOW we can fix empty topics (after fitting)
            try:
                self.logger.info("Calling fix_empty_topics to clean up the model...")
                self.topic_model = self.fix_empty_topics(topic_model)
            except Exception as e:
                self.logger.error(f"Error in fix_empty_topics: continuing with original model:{str(e)}")
            
            # Update topic labels with custom labels
            topic_info, custom_labels = self.save_topic_keywords(topic_model)
            final_model = self.update_topic_labels(topic_info, topic_model)
            
            # Return the trained model
            return final_model

        except Exception as e:
            self.logger.error(f"Error training final model: {str(e)}")
            self.logger.exception(e)
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
        if n_docs > gl.MAX_ADAPTIVE_REPRESENTATIVES:
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
        """
        Generate a topic label from the provided keywords and documents.
        
        Args:
            keywords: List of keywords for the topic
            docs: List of representative documents for the topic
            
        Returns:
            tuple: A tuple containing the topic label and subtopic label
        """
        try:
            # Ensure keywords is a list
            if not keywords:
                return "General Topic", ""
            
            if keywords and len(keywords) > 0:
                # Convert keywords to a list if it's a numpy array
                if isinstance(keywords, np.ndarray):
                    keywords = keywords.tolist()
                
                # Join docs into a single string
                if docs and len(docs) > 0:
                    docs_str = " ".join(docs)
            
            # Process keywords to handle different formats
            processed_keywords = []
            for k in keywords[:5]:
                try:
                    # Handle NumPy arrays first
                    if isinstance(k, np.ndarray):
                        # If k is a numpy array, take the first element
                        if k.size > 0:  # Safe check for array size
                            processed_keywords.append(str(k[0]).replace('_', ' '))
                        else:
                            processed_keywords.append("unknown")
                    # Then handle tuples
                    elif isinstance(k, tuple):
                        if len(k) >= 1:  # This check is now safe
                            processed_keywords.append(str(k[0]).replace('_', ' '))
                        else:
                            processed_keywords.append("unknown")
                    else:
                        # Handle any other type
                        processed_keywords.append(str(k).replace('_', ' '))
                except Exception as e:
                    self.logger.warning(f"Error processing keyword {k}: {e}")
                    processed_keywords.append("unknown")
            
            self.logger.info(f"Generating label for keywords: {processed_keywords}")
            
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
                - Topic: "Business Strategy", Subtopic: "Market Expansion", "Mergers & Acquisitions", "Product Development", "Cost Optimization", "Others"
                - Topic: "Financial Position", Subtopic: "Debt Management", "Liquidity Risk", "Cash Flow", "Working Capital", "Others"
                - Topic: "Corporate Governance", Subtopic: "Board Structure", "Executive Compensation", "Regulatory Compliance", "Others"   
                - Topic: "Technology & Innovation", Subtopic: "Artificial Intelligence", "Digital Transformation", "R&D Investment", "Others"
                - Topic: "Risk Management", Subtopic: "Market Risk", "Operational Risk", "Regulatory Uncertainty", "Financial Stability", "Others"
                - Topic: "Market", Subtopic: "Market Expansion", "Mergers & Acquisitions", "Product Development", "Cost Optimization", "Others"
                - Topic: "Business Overview", Subtopic: "Business Strategy", "Company Description", "Geographic Presence", "Industry Trends", "Market Position", "Product Offerings", "Others"
                - Topic: "Contractual Obligations", Subtopic: "Revenue", "Earnings Per Share", "Gross Margin", "Net Income", "Others"
                - Topic: "Critical Accounting Policies", Subtopic: "Allowance for Doubtful Accounts", "Goodwill Impairment", "Income Taxes", "Inventory Valuation", "Revenue Recognition", "Share-Based Compensation", "Others"
                - Topic: "Financial Performance", Subtopic: "EBITDA", "Earnings Per Share", "Expenses", "Gross Profit", "Net Income", "Operating Income", "Revenues", "Others"
                - Topic: "Forward Looking Statements", Subtopic: "Assumptions", "Future Outlook", "Growth Strategy", "Market Opportunities", "Potential Risks", "Projections", "Others"
                - Topic: "Liquidity and Capital Resources", Subtopic: "Capital Expenditures", "Cash Flow", "Credit Facilities", "Debt Management", "Financing Activities", "Investing Activities", "Working Capital", "Others"
                - Topic: "Off Balance Sheet Arrangements", Subtopic: "Commitments", "Contingent Liabilities", "Guarantees", "Leases", "Variable Interest Entities", "Others"
                - Topic: "Recent Accounting Pronouncements", Subtopic: "Adoption Impact", "Impact Assessment", "Implementation Plans", "New Standards", "Others"
                - Topic: "Recent Developments", Subtopic: "Acquisitions", "Divestitures", "New Products", "Strategic Initiatives", "Others"
                - Topic: "Regulatory and Legal Matters", Subtopic: "Compliance", "Environmental Compliance", "Legal Proceedings", "Legislative Changes", "Regulatory Changes", "Others"
                - Topic: "Risk_Factors", Subtopic: "Competitive Risks", "Economic Conditions", "Financial Risks", "Market Risks", "Operational Risks", "Regulatory Risks", "Others"
                - Topic: "Segment Information", Subtopic: "Geographic Segments", "Product Segments", "Customer Segments", "Segment Performance", "Segment Profitability", "Segment Revenue", "Others"
                - Topic: "Sustainability_and_CSR", Subtopic: "Environmental Impact", "Social Responsibility", "Sustainability Initiatives", "Others"
                - Topic: "Accounting Policies", Subtopic: "Amortization", "Depreciation", "Revenue Recognition", "Income Taxes", "Leases", "Fair Value", "Goodwill"
                - Topic: "Auditor Report", Subtopic: "Audit Opinion", "Critical Audit Matters", "Internal Controls", "Basis for Opinion"
                - Topic: "Cash Flow", Subtopic: "Operating Activities", "Investing Activities", "Financing Activities"
                - Topic: "Corporate Governance", Subtopic: "Board Structure", "Executive Compensation", "Internal Controls", "Strategic Planning"
                - Topic: "Financial Performance", Subtopic: "Revenue", "Operating Income", "Net Income", "EPS", "Segment Results"
                - Topic: "Financial Position", Subtopic: "Assets", "Liabilities", "Equity", "Working Capital", "Investments"
                - Topic: "Business Overview", Subtopic: "Business Model", "Market Position", "Geographic Presence", "Industry Overview"
                - Topic: "Competition", Subtopic: "Market Share", "Competitive Advantages", "Industry Trends"
                - Topic: "Environmental Risks", Subtopic: "Climate Change", "Sustainability", "Resource Management"
                - Topic: "External Factors", Subtopic: "Economic Conditions", "Geopolitical Risks", "Market Conditions"
                - Topic: "Financial Risks", Subtopic: "Credit Risk", "Liquidity Risk", "Interest Rate Risk", "Market Risk"
                - Topic: "Regulatory Matters", Subtopic: "Compliance", "Legal Proceedings", "Regulatory Changes"
                - Topic: "Strategic Initiatives", Subtopic: "Growth Strategy", "Market Expansion", "Innovation"
                - Topic: "Operational Performance", Subtopic: "Efficiency", "Productivity", "Cost Management"
                - Topic: "Market Analysis", Subtopic: "Market Trends", "Consumer Behavior", "Competition"
                - Topic: "Industry Specific Information", Subtopic: "Industry Policy", "Industry Trends", "Regulatory Environment", "Competitive Landscape", "Others"
                ### Keywords:
                - Primary Keywords: {', '.join(processed_keywords)}
                - Secondary Keywords: {', '.join([k.replace('_', ' ') for k in keywords[5:8] if 5 < len(keywords)])}

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
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in firms' fundamental topic analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )

            result = response.choices[0].message.content.strip()
            # Parse the result assuming the output format is "Topic: [Label], Subtopic: [Specific Area]"
            topic_label, subtopic_label = "", ""
            try:
                parts = result.split(",")
                for p in parts:
                    if "Topic" in p:
                        topic_label = p.split(":")[1].strip()
                    elif "Subtopic" in p:
                        subtopic_label = p.split(":")[1].strip()
            except Exception as e:
                self.logger.error(f"Error parsing topic label: {e}")
                topic_label = processed_keywords[0].title() if processed_keywords else "Unlabeled Topic"
                subtopic_label = ' '.join(processed_keywords[1:3]).title() if len(processed_keywords) > 1 else "General"
            
            # Clean up labels
            topic_label = topic_label.replace('"', '').replace("'", "").strip()
            subtopic_label = subtopic_label.replace('"', '').replace("'", "").strip()                                
            
            return topic_label, subtopic_label
            
        except Exception as e:
            self.logger.error(f"Error generating topic label: {e}")
            self.logger.error(traceback.format_exc())
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

    def _generate_topic_labels(self, processed_keywords: list, topic_label: str, subtopic_label: str) -> dict:
        """Generate topic labels for all topics in the topic model."""
        # Add caching to avoid redundant API calls
        cache_key = '_'.join(processed_keywords)  # Create a unique key
        cache_file = os.path.join(gl.output_folder, 'temp', 'topic_label_cache.json')
        
        # Check if we have a cached label
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    label_cache = json.load(f)
                    if cache_key in label_cache:
                        self.logger.info(f"Using cached label for {cache_key}: {label_cache[cache_key]}")
                        return label_cache[cache_key][0], label_cache[cache_key][1]
            else:
                # Initialize empty cache if file doesn't exist
                label_cache = {}
        except Exception as e:
            self.logger.warning(f"Error accessing cache: {e}")
            label_cache = {}  # Use empty cache on error 

        # Cache result for future use
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            cache = {}
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            cache[cache_key] = (topic_label, subtopic_label)  # Store as tuple to avoid parsing issues
            with open(cache_file, 'w') as f:
                json.dump(cache, f)
        except Exception as e:
            self.logger.warning(f"Error caching result: {e}")

    # create a function to load the topic info from json file
    def load_topic_info(self) -> pd.DataFrame:
        """Load topic info from a JSON file."""
        file_path = os.path.join(gl.output_folder, 'temp', 'topic_label_cache.json')
        with open(file_path, 'r') as f:
            return pd.DataFrame(json.load(f))

    def save_topic_keywords(self, topic_model: BERTopic) -> pd.DataFrame:
        """Generate and save topic keywords with labels."""
        try:
            # Create temp directory if it doesn't exist
            temp_dir = os.path.join(gl.output_folder, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Check for cached results first
            cache_file = os.path.join(temp_dir, f'topic_keywords_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{self.n_topics}_{gl.YEAR_START}_{gl.YEAR_END}.pkl')
            if os.path.exists(cache_file):
                self.logger.info("Loading cached topic keywords...")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            # Get basic topic info
            topic_info = topic_model.get_topic_info()
            
            # Add representative documents
            rep_docs = topic_model.representative_docs_ if hasattr(topic_model, "representative_docs_") else {}
            topic_info['Representative_Docs'] = topic_info['Topic'].map(lambda x: rep_docs.get(x, []))
            
            # Initialize lists for labels
            main_topics = [""] * len(topic_info)  # Initialize with empty strings
            subtopics = [""] * len(topic_info)     # Initialize with empty strings
            custom_labels = {}
            # Generate labels (split into topic and subtopic)
            for idx, row in tqdm(topic_info.iterrows(), desc="Generating topic labels"):
                topic_id = row['Topic']
                if topic_id == -1:
                    main_topics[idx] = "No Topic"
                    continue
                    
                try:
                    # Get keywords for the topic
                    topic_keywords = topic_model.get_topics()[topic_id]
                    keywords = []
                    for item in topic_keywords:
                        if isinstance(item, tuple):
                            keywords.append(item[0])
                        else:
                            keywords.append(item)
                    
                    # Get representative documents
                    rep_docs = row['Representative_Docs'] if 'Representative_Docs' in row else []
                    
                    # Generate labels
                    topic_label, subtopic_label = self.generate_topic_label(keywords, rep_docs)
                    main_topics[topic_id] = topic_label
                    subtopics[topic_id] = subtopic_label
                    custom_labels[topic_id] = {
                        "Topic_Label": topic_label,
                        "Subtopic_Label": subtopic_label,
                        "keywords": keywords
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Error generating label for topic {row['Topic']}: {e}")
                    main_topics[idx] = f"Topic {row['Topic']}"
                    subtopics[idx] = "Unclassified"
            
            # Add labels to topic info
            topic_info['Topic_Label'] = main_topics
            topic_info['Subtopic_Label'] = subtopics
            
            # Log the number of topics with labels
            self.logger.info(f"Generated labels for {len(main_topics)} topics")
            
            # Verify labels were generated
            if not any(main_topics):
                self.logger.error("No topic labels were generated")
                raise ValueError("Failed to generate topic labels")
            
            # Save to CSV
            output_path = os.path.join(
                gl.output_folder, 
                f"topic_keywords_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{self.n_topics}_{gl.YEAR_START}_{gl.YEAR_END}.csv"
            )
            topic_info.to_csv(output_path, index=False)
            self.logger.info(f"Saved topic keywords with labels to {output_path}")
            
            # Cache the results
            with open(cache_file, 'wb') as f:
                pickle.dump(topic_info, f)
            self.logger.info(f"Cached topic keywords to {cache_file}")
            
            return topic_info, custom_labels
            
        except Exception as e:
            self.logger.error(f"Error in save_topic_keywords: {e}")
            self.logger.error(traceback.format_exc())
            raise


    def update_topic_labels(self, topic_info: pd.DataFrame, topic_model: BERTopic) -> BERTopic:
        """
        Update topic labels with custom labels.
        
        Args:
            topic_info: DataFrame with topic information
            topic_model: BERTopic model to update
            
        Returns:
            Updated BERTopic model
        """
        try:
            self.logger.info(f"Updating topic labels using DataFrame with columns: {list(topic_info.columns)}")
            # Initialize sub_topic variable
            sub_topic = None
            
            # Create custom labels dictionary
            topic_info, custom_labels = self.save_topic_keywords(topic_model)
            
            # Log custom labels for debugging
            self.logger.info("Custom topic labels:")
            
            # Create a dictionary to store the final formatted labels
            formatted_labels = {}
            
            # Process each topic and create formatted labels
            for topic_id, label in sorted(custom_labels.items()):
                if topic_id >= 0:  # Skip outlier topic (-1)
                    main_topic = label['Topic_Label']
                    sub_topic = label.get('Subtopic_Label', None)  # Use get with default value to avoid KeyError
                    keywords = label.get('keywords', [])
                    
                    # Create a formatted label based on whether sub_topic exists
                    if sub_topic:
                        formatted_label = f"{main_topic} - {sub_topic}"
                    else:
                        formatted_label = f"{main_topic}"
                        
                    # Add to the formatted labels dictionary
                    formatted_labels[topic_id] = formatted_label
                    
                    # Log for debugging
                    self.logger.info(f"  Topic {topic_id}: {formatted_label} (Keywords: {keywords})")
            
            # Update the model with custom labels
            topic_model.set_topic_labels(formatted_labels) 
            self.logger.info(f"Updated topic model with {len(formatted_labels)} custom labels")
            
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
            self.logger.info(f"Debugging topic representatives structure")
            self.logger.info(f"Total topics: {len(topic_representatives)}")
            
            # Check if we have any topics
            if not topic_representatives:
                self.logger.warning("No topics found in topic_representatives")
                return
            
            # Check a sample of topics
            sample_size = min(5, len(topic_representatives))
            sample_topics = list(topic_representatives.items())[:sample_size]
            
            for topic_key, topic_data in sample_topics:
                self.logger.info(f"Topic {topic_key}:")
                self.logger.info(f"  - Number of docs: {len(topic_data.get('docs', []))}")
                self.logger.info(f"  - Number of embeddings: {len(topic_data.get('embeddings', []))}")
                
                # Check if docs and embeddings match
                if len(topic_data.get('docs', [])) != len(topic_data.get('embeddings', [])):
                    self.logger.warning(f"  - Mismatch: {len(topic_data.get('docs', []))} docs vs {len(topic_data.get('embeddings', []))} embeddings")
                
                # Check a sample document
                if topic_data.get('docs'):
                    sample_doc = topic_data['docs'][0]
                    self.logger.info(f"  - Sample doc: {sample_doc[:100]}...")  # First 100 chars
                
                # Check a sample embedding
                if topic_data.get('embeddings'):
                    sample_embedding = topic_data['embeddings'][0]
                    if isinstance(sample_embedding, list):
                        self.logger.info(f"  - Sample embedding: {len(sample_embedding)} dimensions, first 5: {sample_embedding[:5]}")
                    else:
                        self.logger.warning(f"  - Sample embedding is not a list: {type(sample_embedding)}")
            
        except Exception as e:
            self.logger.error(f"Error debugging topic representatives: {e}")
            self.logger.error(traceback.format_exc())

    def _create_topic_model(self, umap_model=None, hdbscan_model=None):
        """
        Create a BERTopic model with the specified parameters.
        
        Args:
            umap_model: UMAP model for dimensionality reduction
            hdbscan_model: HDBSCAN model for clustering
            
        Returns:
            Configured BERTopic model
        """
        try:
            # Log which embedding model we're using
            self.logger.info(f"Creating topic model with embedding model: {self.embedding_model}")
            
            # We don't need to create a SentenceTransformer instance here
            # since we're using pre-computed embeddings
            embedding_model = None
            
            # Get min_df and max_df from global options
            min_df = gl.MIN_DF[0] if hasattr(gl, 'MIN_DF') and gl.MIN_DF else 0.01
            max_df = gl.MAX_DF[0] if hasattr(gl, 'MAX_DF') and gl.MAX_DF else 0.95
            
            # Convert min_df to proportion if it's an absolute count
            if isinstance(min_df, int) and min_df > 1:
                self.logger.info(f"Converting min_df from absolute count ({min_df}) to proportion")
                # We'll estimate the proportion based on a reasonable assumption
                # that we want terms that appear in at least 1% of documents
                min_df = 0.01
            
            # Ensure max_df is a proportion
            if isinstance(max_df, int):
                self.logger.warning(f"max_df was provided as absolute count ({max_df}), converting to proportion")
                max_df = 0.95  # Default to 95% if absolute count was provided
            
            # Ensure min_df is less than max_df
            if min_df >= max_df:
                self.logger.warning(f"min_df ({min_df}) is greater than or equal to max_df ({max_df}). Adjusting parameters.")
                # If min_df is too high, reduce it
                min_df = max(0.001, min_df / 2)  # Ensure at least 0.1% of documents
                # If max_df is too low, increase it
                max_df = min(1.0, max_df * 1.5)
                self.logger.info(f"Adjusted parameters - min_df: {min_df}, max_df: {max_df}")
            
            # Log the final parameters
            self.logger.info(f"Using vectorizer parameters - min_df: {min_df} (proportion), max_df: {max_df} (proportion)")
            
            # Create vectorizer model with validated parameters
            vectorizer_model = CountVectorizer(
                stop_words="english",
                min_df=min_df,
                max_df=max_df,
                ngram_range=(1, 3)  # Add bigrams and tri-grams for better topic modeling
            )
            
            # Get parameters from the models or use defaults
            n_neighbors = getattr(umap_model, 'n_neighbors', self.n_neighbors) if umap_model is not None else self.n_neighbors
            n_components = getattr(umap_model, 'n_components', self.n_components) if umap_model is not None else self.n_components
            min_dist = getattr(umap_model, 'min_dist', self.min_dist) if umap_model is not None else self.min_dist
            
            min_cluster_size = getattr(hdbscan_model, 'min_cluster_size', self.min_cluster_size) if hdbscan_model is not None else self.min_cluster_size
            min_samples = getattr(hdbscan_model, 'min_samples', self.min_samples) if hdbscan_model is not None else self.min_samples
            cluster_selection_epsilon = getattr(hdbscan_model, 'cluster_selection_epsilon', 0.2) if hdbscan_model is not None else 0.2
            
            # If GPU accelerated cuML models are available, use them:
            umap_model, hdbscan_model = _gpu_topic_model(n_neighbors, n_components, min_dist, min_samples, min_cluster_size, cluster_selection_epsilon)
            
            # Create the BERTopic model with the chosen models
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
            self.logger.error(f"Error creating topic model: {e}")
            self.logger.error(traceback.format_exc())
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
                    self.logger.info(f"Saving UMAP embeddings with parameters {params_str} for future use")
                    # Transform embeddings using UMAP
                    umap_embeddings = topic_model.umap_model.transform(embeddings)
                    # Save to file
                    np.save(umap_file, umap_embeddings)
                    self.logger.info(f"UMAP embeddings saved to {umap_file}")
                    return umap_embeddings
                else:
                    self.logger.info(f"UMAP embeddings file already exists at {umap_file}")
        except Exception as e:
            self.logger.warning(f"Error saving UMAP embeddings: {e}")
        return None
        
    def _load_umap_embeddings(self, umap_model):
        """Load pre-computed UMAP embeddings if available."""
        try:
            # Create a filename based on UMAP parameters
            params_str = f"n{umap_model.n_neighbors}_c{umap_model.n_components}_d{umap_model.min_dist:.2f}"
            umap_file = os.path.join(gl.output_folder, f'umap_embeddings_{params_str}.npy')
            
            if os.path.exists(umap_file):
                self.logger.info(f"Loading pre-computed UMAP embeddings from {umap_file}")
                umap_embeddings = np.load(umap_file)
                self.logger.info(f"Loaded UMAP embeddings with shape {umap_embeddings.shape}")
                return umap_embeddings
        except Exception as e:
            self.logger.warning(f"Error loading UMAP embeddings: {e}")
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
            
            self.logger.info(f"Stored representatives for {len(topic_representatives)} topics with a total of {total_docs_stored} documents")
            if skipped_topics > 0:
                self.logger.info(f"Skipped {skipped_topics} topics with fewer than {gl.MIN_DOCS_PER_TOPIC} documents")
            
        except Exception as e:
            self.logger.error(f"Error storing representatives: {e}")
            self.logger.error(traceback.format_exc())
            raise


    def _process_chunks_CPU_parallel(self, docs: list, embeddings: np.ndarray, _cpu = True) -> dict:
        """Process documents in chunks using CPU-optimized parallel processing.
        
        Args:
            docs: List of documents to process
            embeddings: Document embeddings
            _cpu: Flag to indicate CPU processing (default: True)
            
        Returns:
            Dictionary of topic representatives
        """
        try:
            # Calculate optimal chunk size for CPU processing
            chunk_size = self._calculate_optimal_batch_size(for_parallel=True)
            self.logger.info(f"Processing documents in chunks of size {chunk_size} using CPU")
            
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
                chunk_embeddings = embeddings[start_idx:end_idx].copy()
                
                # Add chunk parameters
                chunk_params.append((i, chunk_docs, chunk_embeddings, n_chunks, _cpu))
            
            # Use CPU-optimized process pool
            n_workers = cpu_count()  # Limit to 16 workers for CPU processing
            with Pool(processes=n_workers) as pool:
                results = pool.map(process_chunk_worker, chunk_params)
            
            # Combine results from all workers
            combined_representatives = {}
            for chunk_result in results:
                if chunk_result:  # Only update if the result is not empty
                    combined_representatives.update(chunk_result)
            
            # Debug the combined representatives
            if combined_representatives:
                self._debug_topic_representatives(combined_representatives)
            
            # Save final representatives
            representatives_path = os.path.join(self.output_dirs['models'], 'topic_representatives.json')
            with open(representatives_path, 'w', encoding='utf-8') as f:
                json.dump(combined_representatives, f, ensure_ascii=False, indent=2)
            
            # Log final statistics
            total_topics = len(combined_representatives)
            total_docs = sum(len(topic_data.get('docs', [])) for topic_data in combined_representatives.values())
            self.logger.info(f"CPU parallel processing completed with {total_topics} total topics and {total_docs} representative documents")
            
            return combined_representatives
            
        except Exception as e:
            self.logger.error(f"Error in CPU parallel chunk processing: {e}")
            self.logger.error(traceback.format_exc())
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

    def _calculate_optimal_batch_size(self, for_parallel=False) -> int:
        """
        Calculate optimal batch size based on GPU memory and processing mode.
        
        Args:
            for_parallel: Whether the batch size is for parallel processing
            
        Returns:
            Optimal batch size
        """
        try:
            # Default values if detection fails
            batch_size = gl.GPU_BATCH_SIZE  # Initialize batch_size to default value
            
            # If we're running in parallel mode, reduce batch size to avoid memory issues
            if for_parallel:
                batch_size = batch_size // 2  # Use smaller batches for parallel processing
            
            # Check if GPU is available
            if torch.cuda.is_available():
                # Get GPU info
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.logger.info(f"GPU memory: {gpu_memory_gb:.2f} GB")
                
                # Scale batch size based on GPU memory compared to a reference 16GB card
                memory_factor = gpu_memory_gb / 16.0  # Scale relative to a 16GB card
                
                # Adjust batch size by memory factor, but ensure it's not too small
                adjusted_batch_size = max(int(batch_size * memory_factor), 512)
                
                # If we have very little GPU memory, use CPU mode
                if gpu_memory_gb < 4:
                    self.logger.warning("Low GPU memory detected, using conservative batch size")
                    adjusted_batch_size = min(adjusted_batch_size, 512)
                
                batch_size = adjusted_batch_size
                
            else:
                self.logger.info("No GPU detected, using CPU-optimized batch size")
                # For CPU mode, use a smaller batch size
                batch_size = min(batch_size, 1024)
            
            # In parallel mode, ensure batch size is reasonable for each worker
            if for_parallel:
                batch_size = min(batch_size, 2048)  # Cap at 2048 for parallel
            
            # Log the calculated batch size
            self.logger.info(f"Using batch size: {batch_size}")
            return batch_size
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal batch size: {e}")
            self.logger.error(traceback.format_exc())
            # Return a safe default value
            return 1024

    def _calculate_adaptive_parameters(self, docs: list, embeddings: np.ndarray) -> dict:
        """
        Calculate adaptive parameters for topic modeling based on the dataset characteristics.
        
        Args:
            docs: List of documents to analyze
            embeddings: Document embeddings
            
        Returns:
            Dictionary of adaptive parameters
        """
        self.logger.info("Calculating adaptive parameters for earnings call topic modeling...")
            
        if not isinstance(embeddings, np.ndarray):
            try:
                self.logger.warning("Embeddings passed as list to _calculate_adaptive_parameters, converting to numpy array")
                embeddings = np.array(embeddings, dtype=np.float32)
            except Exception as e:
                self.logger.error(f"Error converting embeddings to numpy array: {e}")
                # Provide default embedding_dim if conversion fails
                embedding_dim = 1024  # Default for most transformer models
                self.logger.warning(f"Using default embedding dimension: {embedding_dim}")
        
        # Safely extract embedding dimension
        try:
            embedding_dim = embeddings.shape[1] if embeddings is not None and hasattr(embeddings, 'shape') and len(embeddings.shape) > 1 else 768
        except (AttributeError, IndexError) as e:
            self.logger.error(f"Error extracting embedding dimension: {e}")
            embedding_dim = 1024  # Default for most transformer models
            self.logger.warning(f"Using default embedding dimension: {embedding_dim}")
        
        # Base parameter calculation
        corpus_size = len(docs)
        self.logger.info(f"Adapting parameters for corpus size: {corpus_size} documents")
        
        # Document length analysis
        avg_doc_length = sum(len(doc.split()) for doc in docs) / corpus_size if corpus_size > 0 else 0
        
        # Detect quarterly patterns and industry clusters
        has_quarterly_patterns = self._detect_quarterly_patterns(docs)
        industry_clusters = self._detect_industry_clusters(docs)
        
        # ADJUSTED: Scale HDBSCAN parameters to generate more topics (around 100)
        # Significantly reduce min_cluster_size for more granular topics
        if corpus_size < 1000:
            min_cluster_size = max(3, int(corpus_size * 0.01))  # Reduced from 0.03
            min_samples = max(5, min_cluster_size // 100)  # More aggressive
        elif corpus_size < 10000:
            min_cluster_size = max(5, int(corpus_size * 0.005))  # Reduced from 0.015
            min_samples = max(5, min_cluster_size // 1000)  # More aggressive
        else:
            # For large corpora, use a much smaller scaling factor
            min_cluster_size = max(50, int(corpus_size * 0.0002))  # Reduced from 0.005
            min_samples = max(5, min_cluster_size // 20)  # More aggressive
        
        # Cap the min_cluster_size to prevent it from being too large
        # This helps ensure we get approximately 100 topics
        min_cluster_size = min(min_cluster_size, 80)  # make it ranges from 50 to 80
        min_samples = min(min_samples, 10)  # Cap at 10, ranges from 5 to 10
        
        # ADJUSTED: Scale UMAP parameters based on embedding dimensionality
        # Increase n_components for better topic separation
        if embedding_dim <= 384:  # Small embedding models
            n_components = max(25, min(100, embedding_dim // 8))  # Increased from // 4
            n_neighbors = max(10, min(25, corpus_size // 800))  # Smaller for finer structure
        elif embedding_dim <= 768:  # Medium embedding models
            n_components = max(30, min(150, embedding_dim // 16))  # Increased from // 8
            n_neighbors = max(15, min(30, corpus_size // 600))  # Smaller for finer structure
        else:  # Large embedding models
            n_components = max(35, min(50, embedding_dim // 32))  # Increased from // 10
            n_neighbors = max(15, min(35, corpus_size // 500))  # Smaller for finer structure
        
        # Adjust for quarterly patterns
        if has_quarterly_patterns:
            self.logger.info("Detected quarterly reporting patterns, adjusting parameters")
            # ADJUSTED: Less aggressive increase to allow more topics
            min_cluster_size = int(min_cluster_size * 1.1)  # Reduced from 1.2
            # Allow more components for better separation
            n_components = max(30, int(n_components * 0.9))  # Less reduction
        
        # Adjust for industry diversity
        if len(industry_clusters) > 5:
            self.logger.info(f"Detected diverse industry clusters: {len(industry_clusters)}")
            # ADJUSTED: More aggressive decrease to allow more topics
            min_cluster_size = max(5, int(min_cluster_size * 0.8))  # Increased reduction from 0.9
            # Need more dimensions to separate industries
            n_components = min(250, int(n_components * 1.3))  # Increased from 1.2
        
        # ADJUSTED: Set min_dist based on document length - using smaller values for tighter clusters
        if avg_doc_length < 50:  # Short documents
            min_dist = max(0.05, min(0.3, avg_doc_length // 500))  # Reduced from 0.05 - tighter clustering
        elif avg_doc_length < 200:  # Medium documents
            min_dist = max(0.1, min(0.3, avg_doc_length // 500))  # Reduced from 0.1 - tighter clustering
        else:  # Long documents
            min_dist = min(0.3, avg_doc_length // 500)  # Reduced from 0.3 - tighter clustering
        
        # ADJUSTED: Decrease epsilon for more precise cluster boundaries
        cluster_selection_epsilon = 0.01  # Reduced from 0.2
        if corpus_size > 20000:
            # For very large corpora, keep epsilon lower to generate more topics
            cluster_selection_epsilon = 0.05  # Reduced from 0.25
        
        self.logger.info(f"Optimized parameters for approximately 100 high-quality topics:")
        self.logger.info(f"  - min_cluster_size: {min_cluster_size} (was much higher)")
        self.logger.info(f"  - min_samples: {min_samples} (was much higher)")
        self.logger.info(f"  - cluster_selection_epsilon: {cluster_selection_epsilon} (more precise)")
        self.logger.info(f"  - n_components: {n_components} (higher dimensionality)")
        self.logger.info(f"  - n_neighbors: {n_neighbors} (more local structure)")
        self.logger.info(f"  - min_dist: {min_dist} (tighter clusters)")
                
        # Store parameters as object attribute and return them
        adaptive_parameters = {
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "min_dist": min_dist,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_epsilon": cluster_selection_epsilon,
            "embedding_dim": embedding_dim,
            "corpus_size": corpus_size,
            "avg_doc_length": avg_doc_length,
            "has_quarterly_patterns": has_quarterly_patterns,
            "industry_clusters": len(industry_clusters)
        }
        
        return adaptive_parameters
    
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

    def save_figures(self, topic_model, specific_visualizations=None, top_n_topics=20, include_doc_vis=False):
        """
        Save visualizations for the topic model.
        
        Args:
            topic_model: The BERTopic model to visualize
            specific_visualizations: List of specific visualizations to save (e.g., ["barchart", "hierarchy"])
                                    If None, all visualizations will be saved
            top_n_topics: Maximum number of topics to include in visualizations
            include_doc_vis: Whether to include document embedding visualization
            
        Returns:
            None
        """
        try:
            self.logger.info(f"Saving visualizations for topic model with {len(topic_model.get_topic_info())} topics")
            
            # Skip visualizations if requested
            if hasattr(gl, 'SKIP_VISUALIZATIONS') and gl.SKIP_VISUALIZATIONS:
                self.logger.info("Visualizations are disabled in config. Skipping.")
                return
                
            # Create or retrieve topic labels
            topic_info, custom_labels = self.save_topic_keywords(topic_model)
            
            # Create visualizer
            visualizer = TopicVis(
                topic_model=topic_model,
                custom_labels=custom_labels
            )
            
            # Determine which visualizations to save
            if not specific_visualizations:
                # Save all visualizations
                self.logger.info(f"Saving all visualizations with top_n_topics={top_n_topics}")
                visualizer.save_all_visualizations(top_n_topics=top_n_topics, include_doc_vis=include_doc_vis)
            else:
                # Save specific visualizations
                for vis_type in specific_visualizations:
                    self.logger.info(f"Saving {vis_type} visualization")
                    if vis_type.lower() == "barchart":
                        visualizer.save_barchart(top_n_topics=top_n_topics)
                    elif vis_type.lower() == "hierarchy":
                        visualizer.save_hierarchy()
                    elif vis_type.lower() == "heatmap":
                        visualizer.save_heatmap()
                    elif vis_type.lower() == "distance_map":
                        visualizer.save_distance_map()
                    elif vis_type.lower() == "document_embeddings" and include_doc_vis:
                        visualizer.save_embedding_document()
                    else:
                        self.logger.warning(f"Unknown visualization type: {vis_type}")
            
            self.logger.info("Visualizations saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving figures: {e}")
            self.logger.error(traceback.format_exc())

    def fix_empty_topics(self, topic_model):
        """Fix empty topics in BERTopic model before saving.
        
        This function removes topic IDs that don't have proper word representations
        and ensures the model's internal structures are consistent.
        """
        self.logger.info("Checking for and removing empty topics before saving model...")
        
        # Get all topic IDs (excluding -1 which is outliers)
        topic_info = topic_model.get_topic_info()
        all_topics = [t for t in topic_info["Topic"].tolist() if t != -1]
        
        # Find which topics have actual content
        valid_topics = []
        invalid_topics = []
        for topic_id in all_topics:
            topic_words = topic_model.get_topic(topic_id)
            if topic_words and isinstance(topic_words, list) and len(topic_words) > 0:
                valid_topics.append(topic_id)
            else:
                invalid_topics.append(topic_id)
        
        if not invalid_topics:
            self.logger.info("No empty topics found, model is clean")
            return topic_model
        
        self.logger.info(f"Found {len(invalid_topics)} empty topics to remove: {invalid_topics}")
        
        # Create a topic mapping: old ID -> new ID
        topic_mapping = {-1: -1}  # Keep outliers as -1
        next_id = 0
        
        for old_id in valid_topics:
            topic_mapping[old_id] = next_id
            next_id += 1
        
        # For invalid topics, map them to -1 (outliers)
        for old_id in invalid_topics:
            topic_mapping[old_id] = -1
        
        # Update topic assignments for all documents
        new_topics = [topic_mapping.get(t, -1) for t in topic_model.topics_]
        topic_model.topics_ = np.array(new_topics)
        
        # Update topic sizes
        new_sizes = {}
        for topic, size in topic_model._topic_sizes.items():
            new_topic = topic_mapping.get(topic, -1)
            if new_topic in new_sizes:
                new_sizes[new_topic] += size
            else:
                new_sizes[new_topic] = size
        topic_model._topic_sizes = new_sizes
        
        # Rebuild topic vectors for remaining topics
        if hasattr(topic_model, '_topic_vectors'):
            new_vectors = {}
            for topic, vector in topic_model._topic_vectors.items():
                if topic in topic_mapping and topic_mapping[topic] != -1:
                    new_vectors[topic_mapping[topic]] = vector
            topic_model._topic_vectors = new_vectors
        
        # Make sure to reduce topics with reduced_topics_
        if hasattr(topic_model, 'reduced_topics_'):
            topic_model.reduced_topics_ = np.array([topic_mapping.get(t, -1) for t in topic_model.reduced_topics_])
        
        self.logger.info(f"Topics remapped. Original count: {len(all_topics)}, New count: {len(valid_topics)}")
        return topic_model



