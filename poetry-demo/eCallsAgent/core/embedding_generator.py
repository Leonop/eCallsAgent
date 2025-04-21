"""
Module for generating and managing document embeddings using SentenceTransformer.
"""

import os
import time
import logging
import glob
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import eCallsAgent.config.global_options as gl # global settings
from eCallsAgent.utils.cuda_setup import init_sentence_transformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for documents using SentenceTransformer."""
    def __init__(self, device: str, model_index: int = None):
        self.device = device
        
        if model_index >= len(gl.EMBEDDING_MODELS):
            logger.warning(f"Model index {model_index} out of range. Using default model.")
            model_index = gl.DEFAULT_MODEL_INDEX
            
        model_name = gl.EMBEDDING_MODELS[model_index]     #gl.EMBEDDING_MODELS[model_index]
        logger.info(f"Using embedding model: {model_name}")
        self.model_name = model_name
        self.model_index = model_index
        
        # Load model
        try:
            # Use the common utility function for initializing SentenceTransformer
            self.model = init_sentence_transformer(model_name, device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            # Make sure we're using the correct dimension from MODEL_DIMENSIONS
            if model_name in gl.MODEL_DIMENSIONS and self.embedding_dim != gl.MODEL_DIMENSIONS[model_name]:
                logger.warning(f"Model {model_name} reports dimension {self.embedding_dim} but expected {gl.MODEL_DIMENSIONS[model_name]}")
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            logger.info("Falling back to default model")
            self.model = init_sentence_transformer(gl.EMBEDDING_MODELS[0], device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.model_name = gl.EMBEDDING_MODELS[0]
            self.model_index = 0
        
        # Optimize for GPU memory and speed
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_mem_gb = gpu_props.total_memory / (1024**3)
            
            logger.info(f"GPU: {gpu_props.name} with {gpu_mem_gb:.2f}GB memory")
            
            # Adjust batch size based on model size and GPU memory
            if 'large' in model_name.lower() or 'bge' in model_name.lower() or self.embedding_dim >= 768:
                # For larger models like BGE-large, gte-large, etc.
                if gpu_mem_gb > 35:  # A100 40GB or similar
                    self.base_batch_size = 384
                    logger.info(f"Large model on high-memory GPU. Using batch size: {self.base_batch_size}")
                else:
                    self.base_batch_size = 128
                    logger.info(f"Large model. Using batch size: {self.base_batch_size}")
            elif gpu_mem_gb > 35:  # For smaller models on high memory GPUs
                self.base_batch_size = gl.GPU_BATCH_SIZE
                logger.info(f"High-memory GPU. Using batch size: {self.base_batch_size}")
            else:
                self.base_batch_size = 256
                logger.info(f"Standard GPU. Using batch size: {self.base_batch_size}")
                
            # Use mixed precision for faster computation on capable GPUs
            if gpu_props.major >= 7:  # Volta architecture or newer
                self.model = self.model.half()  # Convert to FP16
                logger.info("Using FP16 precision for faster computation")
                torch.backends.cudnn.benchmark = True
        else:
            logger.warning("CUDA not available. Using default CPU batch size.")
            self.base_batch_size = 64

    def _calculate_optimal_batch_size(self, embedding_dim: int) -> int:
        """Determine an optimal batch size based on GPU memory."""
        if "cuda" in self.device.lower():
            try:
                gpu_props = torch.cuda.get_device_properties(0)
                total_memory = gpu_props.total_memory
                mem_per_doc = embedding_dim * 8  # 8 bytes per float64
                optimal_batch = int((total_memory * 0.75) / mem_per_doc)
                optimal_batch = min(optimal_batch, self.base_batch_size)
                logger.info(f"Optimal batch size based on GPU memory: {optimal_batch}")
                return optimal_batch
            except Exception as e:
                logger.error(f"Error retrieving GPU properties: {e}, using default batch size.")
                return self.base_batch_size
        else:
            return 64  # Default CPU batch size

    def save_embeddings(self, embeddings: np.ndarray, year_start: int, year_end: int) -> None:
        """Save embeddings to disk."""
        try:
            # Create a model-specific filename
            model_key = self.model_name.replace('/', '-').replace(' ', '_')
            output_path = os.path.join(
                gl.embeddings_folder, 
                f'embeddings_{year_start}_{year_end}_{model_key}.npz'
            )
            
            # Save as compressed file
            np.savez_compressed(
                output_path, 
                embeddings=embeddings, 
                model_name=self.model_name,
                model_index=self.model_index,
                embedding_dim=self.embedding_dim
            )
            
            logger.info(f"Embeddings saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")

    def load_embeddings(self, year_start: int, year_end: int, use_mmap: bool = True) -> np.ndarray:
        """Load embeddings from disk."""
        try:
            # Try to load from memory-mapped file first
            if use_mmap and os.path.exists(gl.TEMP_EMBEDDINGS):
                # Get the size from the file
                file_size = os.path.getsize(gl.TEMP_EMBEDDINGS)
                if file_size > 0:
                    # Infer shape from embedding dimension
                    embedding_dim = self.model.get_sentence_embedding_dimension()
                    shape = (file_size // (embedding_dim * 8), embedding_dim)  # 8 bytes per float64
                    embeddings = np.memmap(gl.TEMP_EMBEDDINGS, dtype=np.float64, mode='r', shape=shape)
                    logger.info(f"Loaded memory-mapped embeddings with shape {embeddings.shape}")
                    return embeddings
            
            # Try to load saved model-specific embeddings
            model_key = self.model_name.replace('/', '-').replace(' ', '_')
            
            # Look for files matching the pattern with dimensions included
            pattern = os.path.join(
                gl.embeddings_folder, 
                f'embeddings_{year_start}_{year_end}_{model_key}.npz'
            )
            matching_files = glob.glob(pattern)
            
            if matching_files:
                # Use the most recent file if multiple matches exist
                embedding_path = max(matching_files, key=os.path.getmtime)
                logger.info(f"Loading embeddings from {embedding_path}")
                data = np.load(embedding_path, allow_pickle=True)
                embeddings = data['embeddings']
                
                # Log metadata
                if 'model_name' in data:
                    logger.info(f"Embeddings were created with model: {data['model_name']}")
                if 'embedding_dim' in data:
                    logger.info(f"Embedding dimension: {data['embedding_dim']}")
                    
                logger.info(f"Loaded compressed embeddings with shape {embeddings.shape}")
                return embeddings
            
            raise FileNotFoundError(f"No embeddings found for model {model_key} in {gl.embeddings_folder}")
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise

    def generate_embeddings(self, docs: list) -> np.ndarray:
        """Generate embeddings using a memory-mapped array for large datasets."""
        start_time = time.time()
        embedding_dim = self.embedding_dim
        
        # Adjust chunk size based on device
        is_cuda = "cuda" in self.device.lower()
        chunk_size = min(8192, len(docs)) if is_cuda else min(64, len(docs))
        
        logger.info(f"Generating embeddings on {self.device} with batch size {chunk_size}")
        
        # Make sure the embeddings folder exists
        os.makedirs(gl.embeddings_folder, exist_ok=True)
        
        # Clean up any existing temporary files
        if os.path.exists(gl.TEMP_EMBEDDINGS):
            os.remove(gl.TEMP_EMBEDDINGS)
            
        # Create a model-specific temporary file
        model_key = self.model_name.replace('/', '-').replace(' ', '_')
        temp_file = os.path.join(gl.embeddings_folder, f'temp_embeddings_{model_key}.mmap')
        
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        # Create temporary memory-mapped file using float64 dtype
        shape = (len(docs), embedding_dim)
        embeddings = np.memmap(temp_file, dtype=np.float64, mode='w+', shape=shape)

        logger.info(f"Generating embeddings for {len(docs)} documents using {self.model_name}")
        logger.info(f"Batch size: {chunk_size}, Embedding dimension: {embedding_dim}")
        
        try:
            # Process in batches
            logger.info("Phase 1: Generating embeddings")
            for i in tqdm(range(0, len(docs), chunk_size), desc="Generating embeddings"):
                end_idx = min(i + chunk_size, len(docs))
                batch = docs[i:end_idx]
                
                # Skip empty documents
                batch = [doc if doc.strip() else "empty document" for doc in batch]
                
                # Generate embeddings with proper error handling
                try:
                    if is_cuda:
                        # Use mixed precision for faster computation on CUDA devices
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                            batch_embed = self.model.encode(
                                batch, 
                                device=self.device,
                                show_progress_bar=False,
                                normalize_embeddings=True
                            )
                    else:
                        batch_embed = self.model.encode(
                            batch, 
                            device=self.device,
                            show_progress_bar=False,
                            normalize_embeddings=True
                        )
                    
                    # Convert batch embeddings to float64 for consistent storage
                    embeddings[i:end_idx] = batch_embed.astype(np.float64)
                    
                    # Flush to disk and clear CUDA cache periodically
                    if i % (chunk_size * 10) == 0 and is_cuda:
                        embeddings.flush()
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Error encoding batch at index {i}: {e}")
                    # Handle error by filling with zeros
                    embeddings[i:end_idx] = np.zeros((end_idx - i, embedding_dim))
            
            total_time = time.time() - start_time
            docs_per_second = len(docs) / total_time
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            logger.info(f"Time: {total_time:.2f} seconds ({docs_per_second:.1f} docs/sec)")
            
            # Copy to standard location for use with topic model
            if os.path.exists(gl.TEMP_EMBEDDINGS):
                os.remove(gl.TEMP_EMBEDDINGS)
                
            # Create a new memory-mapped file at the standard location
            final_embeddings = np.memmap(gl.TEMP_EMBEDDINGS, dtype=np.float64, mode='w+', shape=shape)
            np.copyto(final_embeddings, embeddings)
            final_embeddings.flush()
            
            # Clean up temporary file
            del embeddings
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            # Save the embeddings permanently
            self.save_embeddings(final_embeddings, gl.YEAR_START, gl.YEAR_END)
            logger.info("Embeddings saved for future use")
            
            # Right before the final return statement
            temp_dir = os.path.dirname(gl.TEMP_EMBEDDINGS)
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir, exist_ok=True)
                logger.info(f"Created directory: {temp_dir}")

            # Also verify the file exists before trying to return it
            if not os.path.exists(gl.TEMP_EMBEDDINGS):
                logger.warning(f"Memory-mapped file not found at {gl.TEMP_EMBEDDINGS}, returning from .npz instead")
                # Load from the npz file we just saved
                model_key = self.model_name.replace('/', '-').replace(' ', '_')
                file_path = os.path.join(gl.embeddings_folder, 
                                       f'embeddings_{gl.YEAR_START}_{gl.YEAR_END}.npz')
                saved_data = np.load(file_path)
                return saved_data['embeddings']

            # Now try to return the memory-mapped file
            return np.memmap(gl.TEMP_EMBEDDINGS, dtype=np.float64, mode='r', shape=shape)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise
