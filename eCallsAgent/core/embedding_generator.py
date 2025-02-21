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

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for documents using SentenceTransformer."""
    def __init__(self, device: str):
        self.device = device
        self.model = SentenceTransformer(gl.EMBEDDING_MODELS[0], device=device)
        
        # Optimize for GPU memory and speed
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            
            # Optimize batch size and enable mixed precision
            if 'V100' in gpu_props.name:
                self.base_batch_size = gl.GPU_BATCH_SIZE
                self.model.half()  # Convert to FP16 for faster computation
                torch.backends.cudnn.benchmark = True
            else:
                logger.warning("GPU memory unknown; using default batch size.")
                self.base_batch_size = 128
        else:
            logger.warning("CUDA not available. Using default CPU batch size.")
            self.base_batch_size = 64

    def _calculate_optimal_batch_size(self, embedding_dim: int, default_batch_size: int = gl.GPU_BATCH_SIZE) -> int:
        """Determine an optimal batch size based on GPU memory."""
        if "cuda" in self.device.lower():
            try:
                gpu_props = torch.cuda.get_device_properties(0)
                total_memory = gpu_props.total_memory
                mem_per_doc = embedding_dim * 4  # 4 bytes per float32
                optimal_batch = int((total_memory * 0.75) / mem_per_doc)
                optimal_batch = min(optimal_batch, gl.base_batch_size)
                logger.info(f"Optimal batch size based on GPU memory: {optimal_batch}")
                return optimal_batch
            except Exception as e:
                logger.error(f"Error retrieving GPU properties: {e}, using default batch size.")
                return default_batch_size
        else:
            logger.warning("CUDA not available. Using default CPU batch size.")
            return default_batch_size

    def save_embeddings(self, embeddings: np.ndarray, year_start: int, year_end: int) -> None:
        """Save embeddings in compressed format."""
        filename = f'{gl.data_filename}_embeddings_{year_start}_{year_end}.npz'
        save_path = os.path.join(gl.temp_folder, filename)
        
        # Save in compressed format
        np.savez_compressed(save_path, embeddings=embeddings)
        logger.info(f"Saved compressed embeddings to {save_path}")
        
        # Create backup
        backup_path = save_path.replace('.npz', '_backup.npz')
        np.savez_compressed(backup_path, embeddings=embeddings)
        logger.info(f"Saved compressed backup embeddings to {backup_path}")

    def load_embeddings(self, year_start: int, year_end: int) -> np.ndarray:
        """Load embeddings from compressed format."""
        filename = f'{gl.data_filename}_embeddings_{year_start}_{year_end}.npz'
        load_path = os.path.join(gl.temp_folder, filename)
        
        try:
            with np.load(load_path) as data:
                embeddings = data['embeddings']
            logger.info(f"Loaded embeddings from {load_path}")
            return embeddings
        except FileNotFoundError:
            # Try backup
            backup_path = load_path.replace('.npz', '_backup.npz')
            with np.load(backup_path) as data:
                embeddings = data['embeddings']
            logger.info(f"Loaded embeddings from backup {backup_path}")
            return embeddings

    def generate_embeddings(self, docs: list) -> np.ndarray:
        """Generate embeddings using a memory-mapped array for large datasets."""
        start_time = time.time()
        embedding_dim = self.model.get_sentence_embedding_dimension()
        batch_size = self._calculate_optimal_batch_size(embedding_dim)
        
        # Create temporary memory-mapped file using the standardized path
        shape = (len(docs), embedding_dim)
        embeddings = np.memmap(gl.TEMP_EMBEDDINGS, dtype=np.float32, mode='w+', shape=shape)

        is_cuda = torch.cuda.is_available()
        logger.info(f"Generating embeddings for {len(docs)} documents with batch size {batch_size}")
        
        try:
            for i in tqdm(range(0, len(docs), batch_size), desc="Generating embeddings"):
                end_idx = min(i + batch_size, len(docs))
                batch = docs[i:end_idx]
                if is_cuda:
                    with torch.amp.autocast(device_type='cuda'):
                        batch_embed = self.model.encode(
                            batch, 
                            device=self.device,
                            show_progress_bar=False,
                            normalize_embeddings=True
                        )
                else:
                    batch_embed = self.model.encode(
                        batch, device=self.device,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                embeddings[i:end_idx] = batch_embed
                if i % (batch_size * 5) == 0 and is_cuda:
                    embeddings.flush()
                    torch.cuda.empty_cache()
            
            logger.info(f"Generated embeddings shape: {embeddings.shape} in {time.time() - start_time:.2f} seconds")
            
            # Save embeddings for later use
            self.save_embeddings(embeddings, gl.YEAR_START, gl.YEAR_END)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            try:
                os.unlink(gl.TEMP_EMBEDDINGS)
            except:
                pass
            raise
