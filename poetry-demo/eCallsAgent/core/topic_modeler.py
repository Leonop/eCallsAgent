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
from eCallsAgent.core.visualization import TopicVis
from eCallsAgent.core.chunking_utils import process_chunk_worker, CUML_AVAILABLE  # Add CUML_AVAILABLE import
from eCallsAgent.utils.cuda_setup import setup_cuda, check_cuml_availability 
from eCallsAgent.config import global_options as gl # global settings
import json
import pickle
import gc
import pandas as pd
import csv
import matplotlib.pyplot as plt
import sys  # Added for system debugging
from openai import OpenAI
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Union, Optional
from collections import Counter
import random
from datetime import datetime
import plotly.io as pio
from multiprocessing import Pool, cpu_count
import importlib.util
import subprocess

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
            N_NEIGHBORS, N_COMPONENTS, MIN_DIST,
            MIN_SAMPLES, MIN_CLUSTER_SIZE, 
            NR_TOPICS, METRIC, TOP_N_WORDS,
            MAX_DF, MIN_DF,
            MIN_DOCS_PER_TOPIC, MAX_DOCS_PER_TOPIC,
            MAX_ADAPTIVE_REPRESENTATIVES,
            SEED_TOPICS
        )
        self.cuda_ready = CUDA_READY
        self.cuda_device = CUDA_DEVICE
        self.cuda_memory = CUDA_MEMORY
        
        # Set embedding model selection
        self.pre_trained_model_name = "BAAI/bge-large-en-v1.5"
        self.embedding_model = SentenceTransformer(self.pre_trained_model_name)
        # Store seed topics
        self.seed_topics = SEED_TOPICS        # Store parameters for UMAP 
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
        
        # Add default value for n_topics
        self.n_topics = gl.NR_TOPICS[0]
        
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

    def _transform_documents_gpus(self, docs: list, embeddings: np.ndarray, chunk_size: int = 5000) -> list:
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
                return self.topic_model.transform(docs, embeddings)
            
            gpus_per_node = min(4, num_gpus)  # Use at most 4 GPUs per node
            self.logger.info(f"Using {gpus_per_node} GPUs for document transformation")
            
            # Initialize results collection
            all_topics = []
            all_probs = []
            
            # Split data across GPUs
            docs_per_gpu = len(docs) // gpus_per_node
            remainder = len(docs) % gpus_per_node
            
            # Convert embeddings to appropriate format
            if torch.is_tensor(embeddings):
                gpu_embeddings = embeddings.cuda()
            else:
                gpu_embeddings = embeddings.cpu().numpy()
            
            # Process on each GPU
            for gpu_id in range(gpus_per_node):
                # Calculate start and end indices for this GPU
                start_idx = gpu_id * docs_per_gpu + min(gpu_id, remainder)
                end_idx = start_idx + docs_per_gpu + (1 if gpu_id < remainder else 0)
                
                gpu_topics = []
                gpu_probs = []
                
                # Process in chunks to avoid OOM
                for chunk_start in tqdm(range(start_idx, end_idx, chunk_size),
                                      desc=f"Processing GPU {gpu_id}"):
                    chunk_end = min(chunk_start + chunk_size, end_idx)
                    
                    # Get chunk data
                    chunk_embeddings = gpu_embeddings[chunk_start:chunk_end]
                    
                    try:
                        self.logger.info(f"Processing chunk {chunk_start}-{chunk_end} on GPU {gpu_id}")
                        with torch.cuda.device(gpu_id):
                            topics, probs = self.topic_model.transform(
                                docs[chunk_start:chunk_end], 
                                embeddings[chunk_start:chunk_end]
                            )
                        gpu_topics.extend(topics)
                        gpu_probs.extend(probs)
                            
                    except Exception as e:
                        self.logger.error(f"Error processing chunk {chunk_start}-{chunk_end}: {str(e)}")
                        # Assign noise label (-1) to failed chunks
                        gpu_topics.extend([-1] * (chunk_end - chunk_start))
                        gpu_probs.extend([[0.0]] * (chunk_end - chunk_start))
                
                # Add GPU results to overall results
                all_topics.extend(gpu_topics)
                all_probs.extend(gpu_probs)
                
                self.logger.info(f"Completed processing on GPU {gpu_id}, {len(gpu_topics)} documents")
            
            # Verification step
            if len(all_topics) != len(docs):
                self.logger.warning(f"Mismatch: {len(all_topics)} results for {len(docs)} documents")
                # Handle the mismatch by padding or trimming
                if len(all_topics) < len(docs):
                    all_topics.extend([-1] * (len(docs) - len(all_topics)))
                    all_probs.extend([[0.0]] * (len(docs) - len(all_probs)))
                else:
                    all_topics = all_topics[:len(docs)]
                    all_probs = all_probs[:len(docs)]
            
            self.logger.info(f"Completed transformation of {len(all_topics)} documents")
            return all_topics, all_probs
            
        except Exception as e:
            self.logger.error(f"Error in document transformation: {e}")
            self.logger.error(traceback.format_exc())
            # Return consistent format in error case
            return [-1] * len(docs), [[0.0]] * len(docs)

    def train_topic_model(self, docs: list, embeddings: list) -> BERTopic:
        """
        Train a topic model on the given documents and embeddings.
        
        Args:
            docs: List of documents to model
            embeddings: List of embeddings for the documents
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
                target_topics = min(20000, len(sorted_topics))
                self.logger.info(f"Very large corpus: using up to {target_topics} topics")
            elif len(docs) > 100000:
                target_topics = min(5000, len(sorted_topics))
                self.logger.info(f"Large corpus: using up to {target_topics} topics")
            elif len(docs) > 50000:
                target_topics = min(2500, len(sorted_topics))
                self.logger.info(f"Medium corpus: using up to {target_topics} topics")
            else:
                target_topics = min(2000, len(sorted_topics))
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
                
                for topic in tqdm(sorted_topics[:target_topics], desc="Distilling topics", unit="topic", colour="green"):
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
                    
                    # self.logger.info(f"Topic {topic} has {topic_count} docs (max allowed: {max_docs}, original size: {topic_size})")
                
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
                # Ensure all embeddings have the same shape before conversion
                if len(all_rep_embeddings) > 0:
                    first_shape = np.array(all_rep_embeddings[0]).shape
                    self.logger.info(f"Expected embedding shape: {first_shape}")
                    
                    # Check for consistency
                    for i, emb in enumerate(all_rep_embeddings):
                        emb_array = np.array(emb)
                        if emb_array.shape != first_shape:
                            self.logger.warning(f"Inconsistent embedding at index {i}: {emb_array.shape} vs {first_shape}")
                            # Reshape if possible (same number of elements)
                            if emb_array.size == np.prod(first_shape):
                                all_rep_embeddings[i] = emb_array.reshape(first_shape)
            
                # Convert to numpy array after ensuring consistency
                all_rep_embeddings = np.array(all_rep_embeddings, dtype=np.float32)
            
            self.logger.info(f"Embeddings converted, shape: {all_rep_embeddings.shape}")

            # Get adaptive parameters
            # self.adaptive_parameters = self._calculate_adaptive_parameters(all_rep_docs, all_rep_embeddings)
            
            # Use adaptive parameters if available
            n_neighbors = gl.final_parameters["n_neighbors"]
            n_components = gl.final_parameters["n_components"]
            min_dist = gl.final_parameters["min_dist"]
            min_cluster_size = gl.final_parameters["min_cluster_size"]
            min_samples = gl.final_parameters["min_samples"]
            cluster_selection_epsilon = gl.final_parameters["cluster_selection_epsilon"]
            
       
            self.logger.info(f"Using default parameters (adaptive parameters not available):")
            
            self.logger.info(f"  - n_neighbors: {n_neighbors}")
            self.logger.info(f"  - n_components: {n_components}")
            self.logger.info(f"  - min_dist: {min_dist}")
            self.logger.info(f"  - min_cluster_size: {min_cluster_size}")
            self.logger.info(f"  - min_samples: {min_samples}")
            self.logger.info(f"  - cluster_selection_epsilon: {cluster_selection_epsilon}")
                        
            # Instead of directly calling _gpu_topic_model, check if GPU acceleration is available
            
            self.logger.info("Attempting to create GPU-accelerated models for topic modeling")
            umap_model, hdbscan_model = _gpu_topic_model(n_neighbors, n_components, min_dist, min_samples, min_cluster_size, cluster_selection_epsilon)
            self.logger.info(f"Created topic model with GPU-accelerated cuML and HDBSCAN model")

            # Create the topic model with or without seed topics
            topic_model = self._create_topic_model(umap_model, hdbscan_model)
            topic_model.verbose = False
            
            # Fit the model
            self.logger.info("Fitting topic model to representative documents...")
            topics, probs = topic_model.fit_transform(all_rep_docs, all_rep_embeddings)
            
            # Store the results in instance variables
            self.rep_topics = topics
            self.rep_probs = probs
            self.topic_model = topic_model 
            
            # NOW we can fix empty topics (after fitting)
            try:
                self.logger.info("Calling fix_empty_topics to clean up the model...")
                self.topic_model = self.fix_empty_topics(topic_model)
            except Exception as e:
                self.logger.error(f"Error in fix_empty_topics: continuing with original model:{str(e)}")
            
            # Update topic labels with custom labels
            topic_info, _ = self.save_topic_keywords(topic_model)
            final_model = self.update_topic_labels(topic_info, topic_model)
            
            # Return the trained model
            return final_model
        except Exception as e:
            self.logger.error(f"Error training final model: {str(e)}")
            self.logger.exception(e)
            raise

    def _map_documents(self, docs: list, embeddings: np.ndarray) -> tuple:
        """
        Map all documents to topics and probabilities of topic assignments using the trained model.
        
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
                self.logger.info("Using GPU for document mapping and generate topics and probabilities")
                topics, probs = self._transform_documents_gpus(docs, embeddings)
                
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
            for topic in tqdm(sorted_topics, desc="Collecting topic representatives", unit="topic", colour="green"):
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
            tuple: A tuple containing the topic label, subtopic label, confidence score
        """
        try:
            # Ensure keywords is a list
            if not keywords:
                return "General Topic", "Generic", 0.0
            
            if keywords and len(keywords) > 0:
                # Convert keywords to a list if it's a numpy array
                if isinstance(keywords, np.ndarray):
                    keywords = keywords.tolist()
                
                # Join docs into a single string
                if docs and len(docs) > 0:
                    docs_str = " ".join(docs)
                
            # Process keywords to handle different formats
            processed_keywords = []
            for k in keywords:
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
            
            # self.logger.info(f"Generating label for keywords: {processed_keywords}")
            
            # Make sure we have at least one keyword
            if not processed_keywords:
                return "Unlabeled Topic", "General", 0.0
            
            # Safely check the cached labels
            try:
                # Check if we already have a cached label
                cached_result = self._load_topic_labels(processed_keywords)
                if cached_result and isinstance(cached_result, tuple) and len(cached_result) == 3:
                    cached_topic, cached_subtopic, cached_confidence = cached_result
                    if cached_topic != "Unlabeled Topic":  # Only use cache if we have a valid label
                        return cached_topic, cached_subtopic, cached_confidence
                else:
                    # If _load_topic_labels returned None or invalid format
                    self.logger.warning("Cache returned invalid format, using fallback labels")
            except Exception as e:
                self.logger.warning(f"Error checking cache: {e}")
                # Continue with label generation
            
            # Check if OpenAI is available; if not, use a simple fallback
            if not hasattr(self, 'openai_available') or not self.openai_available or not hasattr(self, 'client'):
                self.logger.warning("OpenAI not available. Using fallback labeling.")
                main_topic = processed_keywords[0].title() if processed_keywords else "Unlabeled Topic"
                subtopic = ' '.join(processed_keywords[1:3]).title() if len(processed_keywords) > 1 else "General"
                return main_topic, subtopic, 0.0
            
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
                - Output format: Topic: [Label], Subtopic: [Specific Area], Confidence: [0-1]
                - Be concise and specific."""

            try:
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
                topic_label, subtopic_label, confidence = "", "", 0.0
                try:
                    parts = result.split(",")
                    for p in parts:
                        if "Topic" in p:
                            topic_label = p.split(":")[1].strip()
                        elif "Subtopic" in p:
                            subtopic_label = p.split(":")[1].strip()
                        elif "Confidence" in p:
                            try:
                                confidence = float(p.split(":")[1].strip())
                            except (ValueError, IndexError):
                                self.logger.warning(f"Could not parse confidence value: {p}")
                                confidence = 0.0
                    
                    # Clean up labels
                    topic_label = topic_label.replace('"', '').replace("'", "").strip()
                    subtopic_label = subtopic_label.replace('"', '').replace("'", "").strip()
                    
                    # If we couldn't parse properly, use fallback
                    if not topic_label:
                        topic_label = processed_keywords[0].title()
                    if not subtopic_label:
                        subtopic_label = "General"
                    
                    return topic_label, subtopic_label, confidence
                except Exception as e:
                    self.logger.error(f"Error parsing topic label: {e}")
                    # Fallback to simple labeling
                    topic_label = processed_keywords[0].title() if processed_keywords else "Unlabeled Topic"
                    subtopic_label = ' '.join(processed_keywords[1:3]).title() if len(processed_keywords) > 1 else "General"
                    return topic_label, subtopic_label, 0.0
            except Exception as e:
                self.logger.error(f"Error calling OpenAI API: {e}")
                # Fallback to simple labeling
                topic_label = processed_keywords[0].title() if processed_keywords else "Unlabeled Topic"
                subtopic_label = ' '.join(processed_keywords[1:3]).title() if len(processed_keywords) > 1 else "General"
                return topic_label, subtopic_label, 0.0
                    
        except Exception as e:
            self.logger.error(f"Error generating topic label: {e}")
            self.logger.error(traceback.format_exc())
            # Always return a valid tuple even in case of errors
            if processed_keywords and len(processed_keywords) > 0:
                return processed_keywords[0].title(), "General", 0.0
            return "Unlabeled Topic", "General", 0.0

    def _load_topic_labels(self, processed_keywords: list) -> tuple:
        """
        Load topic labels from cache if they exist.
        
        Args:
            processed_keywords: List of keywords to use as cache key
            
        Returns:
            Tuple of (topic_label, subtopic_label) or ("Unlabeled Topic", "General") if not found
        """
        # If no keywords, return default
        if not processed_keywords:
            return "Unlabeled Topic", "General", 0.0
            
        # Add caching to avoid redundant API calls
        try:
            cache_key = '_'.join(processed_keywords)  # Create a unique key
            cache_file = os.path.join(gl.output_folder, 'temp', f'topic_label_cache_{gl.YEAR_START}_{gl.YEAR_END}.json')
            
            # Create temp directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            # Check if we have a cached label
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        label_cache = json.load(f)
                        if cache_key in label_cache:
                            cached_data = label_cache[cache_key]
                            if isinstance(cached_data, list) and len(cached_data) >= 2:
                                self.logger.info(f"Using cached label for {cache_key}: {cached_data}")
                                # Make sure we return all three expected values
                                if len(cached_data) == 2:
                                    # Handle legacy cache format (before confidence was added)
                                    return cached_data[0], cached_data[1], 0.0
                                elif len(cached_data) >= 3:
                                    # Current format with confidence score
                                    return cached_data[0], cached_data[1], cached_data[2]
                            else:
                                self.logger.warning(f"Invalid cache format for {cache_key}: {cached_data}")
                except json.JSONDecodeError:
                    self.logger.warning(f"Error decoding JSON from cache file: {cache_file}")
                    # Create a new empty cache file
                    with open(cache_file, 'w') as f:
                        json.dump({}, f)
            else:
                # Initialize empty cache if file doesn't exist
                self.logger.info(f"Creating new cache file: {cache_file}")
                with open(cache_file, 'w') as f:
                    json.dump({}, f)
            
            # No valid cached label found - return default values
            return "Unlabeled Topic", "General", 0.0
        
        except Exception as e:
            self.logger.warning(f"Error accessing cache: {e}")
            self.logger.warning(traceback.format_exc())
            return "Unlabeled Topic", "General", 0.0


    def save_topic_keywords(self, topic_model: BERTopic) -> tuple[pd.DataFrame, dict]:
        """Generate and save topic keywords with labels.
        and save the topic labels in a cache file"""
        try:
            # Create temp directory if it doesn't exist
            temp_dir = os.path.join(gl.output_folder, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Initialize cache file for topic labels
            cache_file = os.path.join(temp_dir, f'topic_label_cache_{gl.YEAR_START}_{gl.YEAR_END}.json')
            if not os.path.exists(cache_file):
                with open(cache_file, 'w') as f:
                    json.dump({}, f)
            
            # Try to load existing cache
            try:
                with open(cache_file, 'r') as f:
                    label_cache = json.load(f)
            except:
                label_cache = {}
                
            # Get basic topic info
            topic_info = topic_model.get_topic_info()
            
            # Add representative documents
            rep_docs = topic_model.representative_docs_ if hasattr(topic_model, "representative_docs_") else {}
            topic_info['Representative_Docs'] = topic_info['Topic'].map(lambda x: rep_docs.get(x, []))
            
            # Initialize lists for labels
            num_topics = len(topic_info)
            main_topics = [""] * num_topics
            subtopics = [""] * num_topics
            custom_labels = {}
            
            # Generate labels (split into topic and subtopic)
            for idx, row in tqdm(topic_info.iterrows(), desc="Generating topic labels"):
                topic_id = row['Topic']
                if topic_id == -1:
                    main_topics[idx] = "No Topic"
                    subtopics[idx] = "General"
                    continue
                    
                try:
                    # Get keywords for the topic
                    topic_keywords = topic_model.get_topics()[topic_id]
                    keywords = []
                    
                    # Extract keywords
                    for item in topic_keywords[:5]:  # Limit to top 5 keywords
                        if isinstance(item, tuple):
                            keywords.append(item[0])
                        else:
                            keywords.append(item)
                        
                    # Get representative documents
                    rep_docs = row['Representative_Docs'] if 'Representative_Docs' in row else []
                    
                    # Generate labels with error handling
                    try:
                        topic_label, subtopic_label, confidence = self.generate_topic_label(keywords, rep_docs)
                        
                        # Verify we got valid labels
                        if not topic_label or not isinstance(topic_label, str):
                            topic_label = f"Topic {topic_id}"
                        if not subtopic_label or not isinstance(subtopic_label, str):
                            subtopic_label = "General"
                            
                    except Exception as e:
                        self.logger.warning(f"Error generating label for topic {topic_id}: {e}")
                        # Use default labels if topic generation fails
                        topic_label = f"Topic {topic_id}"
                        subtopic_label = "General"
                        if keywords:
                            # Use the first keyword as topic label if available
                            topic_label = keywords[0].title()
                    
                    # Update the cache with new labels
                    cache_key = '_'.join(keywords)
                    label_cache[cache_key] = [topic_label, subtopic_label, confidence]
                    
                    # Store the labels
                    main_topics[idx] = topic_label
                    subtopics[idx] = subtopic_label
                    custom_labels[topic_id] = {
                        "Topic_Label": topic_label,
                        "Subtopic_Label": subtopic_label,
                        "Confidence": confidence,
                        "keywords": keywords
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Error generating label for topic {row['Topic']}: {e}")
                    self.logger.warning(traceback.format_exc())
                    main_topics[idx] = f"Topic {row['Topic']}"
                    subtopics[idx] = "Unclassified"
            
            # Add labels to topic info
            topic_info['Topic_Label'] = main_topics
            topic_info['Subtopic_Label'] = subtopics
            topic_info['Confidence'] = [custom_labels.get(topic, {}).get('Confidence', 0.0) for topic in topic_info['Topic']]
            # Save updated cache
            try:
                with open(cache_file, 'w') as f:
                    json.dump(label_cache, f, indent=2)
            except Exception as e:
                self.logger.warning(f"Error saving topic label cache: {e}")
            
            # Log the number of topics with labels
            self.logger.info(f"Generated labels for {len(main_topics)} topics")

            # Verify labels were generated
            if not any(main_topics):
                self.logger.error("No topic labels were generated")
                raise ValueError("Failed to generate topic labels")
            
            # Save to CSV
            output_path = os.path.join(
                gl.output_folder, 
                f"topic_keywords_{gl.final_parameters['n_neighbors']}_{gl.final_parameters['n_components']}_{gl.final_parameters['min_cluster_size']}_{self.n_topics}_{gl.YEAR_START}_{gl.YEAR_END}.csv"
            )
            topic_info.to_csv(output_path, index=False)
            self.logger.info(f"Saved topic keywords with labels to {output_path}")

            # check if the custom labels file exists
            custom_labels_file = os.path.join(gl.output_folder, 'temp', 'custom_labels.json')
            # Ensure the temp directory exists
            os.makedirs(os.path.dirname(custom_labels_file), exist_ok=True)
            
            if not os.path.exists(os.path.join(gl.output_folder, 'temp', 'acc_cache')):
                os.makedirs(os.path.join(gl.output_folder, 'temp', 'acc_cache'))
            acc_cache_file = os.path.join(gl.output_folder, 'temp', 'acc_cache', f'topic_label_cache_{gl.YEAR_START}_{gl.YEAR_END}.json')

            if os.path.exists(custom_labels_file):
                # append the custom labels to the cache file
                with open(custom_labels_file, 'r') as f:
                    label_cache = json.load(f)
                    label_cache.update(custom_labels)
                with open(custom_labels_file, 'w') as f:
                    json.dump(label_cache, f)
                # Save to acc_cache file separately
                with open(acc_cache_file, 'w') as f:
                    json.dump(label_cache, f)
            else:
                # dump the custom labels to the cache file
                with open(custom_labels_file, 'w') as f:
                    json.dump(custom_labels, f)
                # Save to acc_cache file separately  
                with open(acc_cache_file, 'w') as f:
                    json.dump(custom_labels, f)
            self.logger.info(f"Cached topic keywords to {custom_labels_file}")
            
            # Always return both topic_info and custom_labels
            return topic_info, custom_labels
            
        except Exception as e:
            self.logger.error(f"Error saving topic keywords: {e}")
            self.logger.error(traceback.format_exc())
            # Return empty dataframe and dict to avoid None
            return pd.DataFrame(), {}


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
            
            # Create a dictionary to store the final formatted labels
            formatted_labels = {}
            
            # Process each topic and create formatted labels
            for _, row in tqdm(topic_info.iterrows(), desc="Updating topic labels", unit="topic", colour="green"):
                topic_id = row['Topic']
                if topic_id >= 0:  # Skip outlier topic (-1)
                    main_topic = row['Topic_Label']
                    sub_topic = row.get('Subtopic_Label', None)  # Use get with default value to avoid KeyError
                    
                    # Create a formatted label based on whether sub_topic exists
                    if sub_topic:
                        formatted_label = f"{main_topic}_{sub_topic}"
                    else:
                        formatted_label = f"{main_topic}_General"
                        
                    # Add to the formatted labels dictionary
                    formatted_labels[topic_id] = formatted_label
                    
                    # Log for debugging
                    self.logger.info(f"  Topic {topic_id}: {formatted_label}")
            
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

    def embed_seed_topics(self):
        """
        Convert seed topics to embeddings using the same model used for documents.
        
        Args:
            seed_topic_list (list): List of lists of keywords representing seed topics
                e.g., [["finance", "earnings", "revenue"], ["technology", "innovation"]]
        
        Returns:
            tuple: (seed_topic_strings, seed_topic_embeddings)
        """
        self.logger.info(f"Embedding {len(gl.SEED_TOPICS)} seed topics with {self.pre_trained_model_name}")
        
        # Convert each topic list to a string by joining keywords
        seed_topic_strings = [" ".join(keywords) for keywords in gl.SEED_TOPICS]
        
        # Use the same SentenceTransformer model to embed the topics
        seed_topic_embeddings = self.embedding_model.encode(
            seed_topic_strings, 
            show_progress_bar=True, 
            convert_to_numpy=True,
            normalize_embeddings=True  # Ensure normalized embeddings
        )

        self.logger.info(f"Created embeddings for {len(seed_topic_strings)} seed topics with shape {seed_topic_embeddings.shape}")
        # save the embeddings to a file
        np.save(os.path.join(gl.output_folder, "seed_topic_embeddings.npy"), seed_topic_embeddings)
        return seed_topic_embeddings

    def _create_topic_model(self, umap_model=None, hdbscan_model=None, _cpu = False):
        """Create a topic model with the specified UMAP and HDBSCAN models."""
        try:
            params = gl.final_parameters
            
            self.logger.info(f"Using parameters for topic model:")
            self.logger.info(f"  n_neighbors: {params['n_neighbors']}")
            self.logger.info(f"  n_components: {params['n_components']}")
            self.logger.info(f"  min_dist: {params['min_dist']}")
            self.logger.info(f"  min_cluster_size: {params['min_cluster_size']}")
            self.logger.info(f"  min_samples: {params['min_samples']}")
            self.logger.info(f"  cluster_selection_epsilon: {params['cluster_selection_epsilon']}")
            
            # Log which embedding model we're using
            self.logger.info(f"Creating topic model with embedding model: {self.embedding_model}")
            
            # Create vectorizer model with validated parameters
            vectorizer_model = CountVectorizer(
                stop_words="english",
                min_df=gl.MIN_DF[0],
                max_df=gl.MAX_DF[0],
                ngram_range=(1, 3)  # Add bigrams and tri-grams for better topic modeling
                )
            # Check if we have seed topics defined
            if hasattr(self, 'seed_topics') and self.seed_topics:
                self.logger.info(f"Processing {len(self.seed_topics)} seed topics for compatibility")

                try:
                    # Create the BERTopic model with the chosen models
                    topic_model = BERTopic(
                    embedding_model=self.embedding_model,
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer_model,
                    nr_topics='auto',
                    seed_topic_list=self.seed_topics,
                    top_n_words=gl.TOP_N_WORDS[0],
                    calculate_probabilities=False,
                    verbose=False
                    )
                except Exception as e:                 # Create the BERTopic model with the chosen models
                    topic_model = BERTopic(
                        embedding_model=self.embedding_model,
                        umap_model=umap_model,
                        hdbscan_model=hdbscan_model,
                        vectorizer_model=vectorizer_model,
                        nr_topics='auto',
                        seed_topic_list=None,
                        top_n_words=gl.TOP_N_WORDS[0],
                        calculate_probabilities=False,
                        verbose=False
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
            
            # Get unique topics for progress bar
            unique_topics = set(topics)
            unique_topics = [t for t in unique_topics if t != -1]  # Remove outlier topic
            
            # Process each topic (except outlier topic -1)
            for topic_id in tqdm(unique_topics, desc="Processing topic representatives", unit="topic", colour="green"):
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
            chunk_size = self._calculate_optimal_batch_size(_cpu=True)  
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
                # Ensure embeddings are on CPU and detached from any CUDA tensors
                if isinstance(embeddings, torch.Tensor):
                    chunk_embeddings = embeddings[start_idx:end_idx].cpu().numpy()
                else:
                    chunk_embeddings = embeddings[start_idx:end_idx].copy()
                
                # Add chunk parameters
                chunk_params.append((i, chunk_docs, chunk_embeddings, n_chunks, 
                                     None,  _cpu))
            
            # Use CPU-optimized process pool with spawn method
            n_workers = min(cpu_count(), 16)  # Limit to 16 workers for CPU processing
            self.logger.info(f"Starting parallel processing with {n_workers} workers for {len(chunk_params)} chunks")
            
            # Clean up any CUDA memory before forking
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            combined_representatives = {}
            with Pool(processes=n_workers) as pool:
                # Process chunks with progress bar
                task_iter = pool.starmap(process_chunk_worker, chunk_params)
                for chunk_result in tqdm(task_iter, total=len(chunk_params), desc="Processing document chunks", unit="chunk", colour='orange'):
                    if chunk_result:  # Only update if the result is not empty
                        combined_representatives.update(chunk_result)
            
            # Debug the combined representatives with a simplified summary
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
            if 'large' in self.pre_trained_model_name or 'bge' in self.pre_trained_model_name:
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

    def _calculate_optimal_batch_size(self, _cpu=True) -> int:
        """Calculate the optimal batch size for document processing based on available resources."""
        try:
            # Start with a reasonable default batch size
            DEFAULT_CPU_BATCH_SIZE = 5000
            DEFAULT_GPU_BATCH_SIZE = 10000
            
            # Determine initial batch size based on CPU or GPU processing
            batch_size = DEFAULT_CPU_BATCH_SIZE if _cpu else DEFAULT_GPU_BATCH_SIZE
            
            # Adjust based on memory availability if using GPU
            if not _cpu and torch.cuda.is_available():
                try:
                    # Get GPU memory
                    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    gpu_mem_available = torch.cuda.memory_reserved(0) / (1024**3)
                    
                    # Assign batch size based on available GPU memory
                    if gpu_mem_total > 16:  # High-end GPUs
                        batch_size = 10000
                    elif gpu_mem_total > 8:  # Mid-range GPUs
                        batch_size = 5000
                    elif gpu_mem_total > 4:  # Entry-level GPUs
                        batch_size = 3000
                    else:  # Very limited GPUs
                        batch_size = 2000
                        
                    self.logger.info(f"GPU memory: {gpu_mem_total:.2f} GB, allocating batch size of {batch_size}, remaining memory: {gpu_mem_available:.2f} GB ")
                    
                except Exception as e:
                    self.logger.warning(f"Error determining GPU memory, using default batch size {batch_size}: {e}")
            
            # For CPU processing, adjust batch size based on available CPU cores
            elif _cpu:
                try:
                    # Get CPU cores and adjust batch size accordingly
                    num_cores = cpu_count()
                    
                    # Smaller batch size for systems with less CPU power
                    if num_cores <= 4:
                        batch_size = 1000
                    elif num_cores <= 8:
                        batch_size = 2000
                    elif num_cores <= 16:
                        batch_size = 3000
                    else:  # High-core systems
                        batch_size = 6000
                        
                    self.logger.info(f"CPU cores: {num_cores}, allocating batch size of {batch_size}")
                    
                except Exception as e:
                    self.logger.warning(f"Error determining CPU cores, using default batch size {batch_size}: {e}")
            
            # Ensure batch size is reasonable and doesn't exceed 10,000
            batch_size = max(1000, min(batch_size, 10000))
            
            return batch_size
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal batch size: {e}")
            # Return a safe default
            return 2000 if _cpu else 4000

    def _calculate_adaptive_parameters(self, docs: list, embeddings: np.ndarray) -> dict:
        """Calculate adaptive parameters for topic modeling based on the dataset characteristics."""
        self.logger.info("Calculating adaptive parameters for earnings call topic modeling...")
            
        embedding_dim = gl.EMBEDDING_DIM
        # Base parameter calculation
        corpus_size = len(docs)
        self.logger.info(f"Adapting parameters for corpus size: {corpus_size} documents")
        
        # Document length analysis
        avg_doc_length = sum(len(doc.split()) for doc in docs[:1000]) / min(corpus_size, 1000) if corpus_size > 0 else 0
        
        # OPTIMIZED: Parameters for less strict clustering to reduce noise points
        if corpus_size < 5000:
            min_cluster_size = 3  # Very small for small datasets
            min_samples = 2
        elif corpus_size < 50000:
            min_cluster_size = 5  # Small for medium datasets
            min_samples = 2
        elif corpus_size < 200000:
            min_cluster_size = 10  # Medium for large datasets
            min_samples = 3
        else:
            min_cluster_size = 15  # Smaller value for very large datasets
            min_samples = 3  # Reduced from 5 to be less strict
        
        # Increase n_components for better topic separation but keep n_neighbors low for more local structure
        if embedding_dim <= 384:  # Small embedding models
            n_components = 75  # Increased for better separation
            n_neighbors = max(8, min(20, corpus_size // 1000))  # Smaller for more local structure
        elif embedding_dim <= 768:  # Medium embedding models
            n_components = 200  # Increased for better separation
            n_neighbors = max(10, min(25, corpus_size // 800))  # Smaller for more local structure
        else:  # Large embedding models (1024 in your case)
            n_components = 400  # Increased from 150 to capture more subtle differences
            n_neighbors = max(12, min(10, corpus_size // 8000))  # Reduced for more local structure
        
        # ADJUSTED: Set min_dist based on document length - using smaller values for tighter clusters
        if avg_doc_length < 50:  # Short documents
            min_dist = 0.0  # Minimum value for tightest clustering
        elif avg_doc_length < 200:  # Medium documents
            min_dist = 0.05  # Very small for tight clustering
        else:  # Long documents
            min_dist = 0.05  # Still relatively small
        
        # ADJUSTED: Increase epsilon for more lenient cluster boundaries
        cluster_selection_epsilon = 0.05  # Increased from 0.05 to be more lenient
        if corpus_size > 20000:
            # For very large corpora, increase epsilon further
            cluster_selection_epsilon = 0.01  # Increased to be even more lenient

        # A100-specific optimizations
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            if "A100" in gpu_props.name:
                # Allow more aggressive parameters for A100
                n_components = min(500, n_components * 1.5)  # Increased for better separation
                min_cluster_size = max(20, min_cluster_size)  # Reduced to allow smaller clusters
            
        self.logger.info(f"Optimized parameters for reducing noise points:")
        self.logger.info(f"  - min_cluster_size: {min_cluster_size}")
        self.logger.info(f"  - min_samples: {min_samples}")
        self.logger.info(f"  - cluster_selection_epsilon: {cluster_selection_epsilon}")
        self.logger.info(f"  - n_components: {n_components}")
        self.logger.info(f"  - n_neighbors: {n_neighbors}")
        self.logger.info(f"  - min_dist: {min_dist}")
        
        # Return parameters
        return {
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "min_dist": min_dist,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_epsilon": cluster_selection_epsilon,
            "embedding_dim": embedding_dim,
            "corpus_size": corpus_size,
            "avg_doc_length": avg_doc_length,
        }
    
    def load_custom_labels(self):
        """Load custom labels from the custom_labels.json file."""
        try:
            with open(os.path.join(gl.output_folder, 'temp', 'custom_labels.json'), 'r') as f:
                custom_labels = json.load(f)
            return custom_labels
        except Exception as e:
            self.logger.error(f"Error loading custom labels: {e}")
            return {}  # Return empty dict instead of None
        
    def convert_custom_labels(self, custom_labels):
        """Convert custom labels to main_topic_subtopic format."""
        try:
            # Create a new dictionary with the desired format
            converted_labels = {}
            for topic_id, label in custom_labels.items():
                # Handle both lowercase and uppercase key variations
                topic_label = label.get('Topic_Label') or label.get('topic_label')
                subtopic_label = label.get('Subtopic_Label') or label.get('subtopic_label')
                
                # Skip if we don't have both labels
                if not topic_label or not subtopic_label:
                    self.logger.warning(f"Missing labels for topic {topic_id}: {label}")
                    continue
                    
                custom_label = f"{topic_label}_{subtopic_label}"
                converted_labels[int(topic_id)] = custom_label
            return converted_labels
        except Exception as e:
            self.logger.error(f"Error converting custom labels: {e}")
            self.logger.error(traceback.format_exc())
            return {}
        
    def confidence_topic_score(self):
        """Calculate the mean confidence score of all topics."""
        # load the custom labels.json file
        custom_labels = self.load_custom_labels()
        confidence_scores = []
        
        # Handle case where custom_labels is empty
        if not custom_labels:
            self.logger.warning("No custom labels found, returning default confidence score of 0.0")
            return 0.0, 0.0
            
        # get the topic confidence score
        for topic_id, label in custom_labels.items():
            # Handle different label formats
            if 'Confidence' in label:
                confidence_score = label['Confidence']
            else:
                self.logger.warning(f"No confidence score found for topic {topic_id}, using default 0.0")
                confidence_score = 0.0
                
            confidence_scores.append(confidence_score)
            
        # Handle case where no confidence scores were found
        if not confidence_scores:
            self.logger.warning("No confidence scores found in custom labels, returning default of 0.0")
            return 0.0, 0.0
            
        # compute the statistical description of the confidence scores
        mean_confidence_score = sum(confidence_scores)/len(confidence_scores)
        std_confidence_score = np.std(confidence_scores)
        return mean_confidence_score, std_confidence_score

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
                
            # load custom labels
            custom_labels = self.load_custom_labels()
            # convert custom labels to main_topic_subtopic format 
            formatted_custom_labels = self.convert_custom_labels(custom_labels)

            # Create visualizer
            visualizer = TopicVis(
                topic_model=topic_model,
                custom_labels=formatted_custom_labels
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