"""
Module for coordinating distributed processing across multiple nodes.
"""

import os
import logging
import torch.distributed as dist
import torch
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

class DistributedCoordinator:
    """Coordinates distributed processing across multiple nodes."""
    
    def __init__(self, n_nodes: int = 5, gpus_per_node: int = 1):
        """Initialize distributed coordinator.
        
        Args:
            n_nodes: Number of nodes available
            gpus_per_node: Number of GPUs per node
        """
        self.n_nodes = n_nodes
        self.gpus_per_node = gpus_per_node
        self.total_gpus = n_nodes * gpus_per_node
        self.node_rank = int(os.environ.get('SLURM_NODEID', 0))
        self.world_size = self.total_gpus
        
        # Initialize process group
        try:
            if not dist.is_initialized():
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=self.world_size,
                    rank=self.node_rank
                )
            logger.info(f"Initialized distributed process group on node {self.node_rank}")
        except Exception as e:
            logger.error(f"Failed to initialize distributed process group: {e}")
            raise
    
    def distribute_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Distribute data across nodes.
        
        Args:
            data: Input data to distribute
            
        Returns:
            Tuple of (node_data, node_indices)
        """
        total_size = len(data)
        chunk_size = total_size // self.world_size
        start_idx = self.node_rank * chunk_size
        end_idx = start_idx + chunk_size if self.node_rank != self.world_size - 1 else total_size
        
        node_data = data[start_idx:end_idx]
        node_indices = np.arange(start_idx, end_idx)
        
        logger.info(f"Node {self.node_rank}: Processing {len(node_data)} items")
        return node_data, node_indices
    
    def gather_results(self, local_results: np.ndarray) -> Optional[np.ndarray]:
        """Gather results from all nodes.
        
        Args:
            local_results: Results from current node
            
        Returns:
            Combined results from all nodes if on main node, None otherwise
        """
        try:
            # Convert to tensor for gathering
            local_tensor = torch.from_numpy(local_results).cuda()
            gathered = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]
            
            dist.all_gather(gathered, local_tensor)
            
            if self.node_rank == 0:
                # Combine results on main node
                combined = torch.cat(gathered).cpu().numpy()
                logger.info(f"Gathered results from all nodes: {combined.shape}")
                return combined
            return None
            
        except Exception as e:
            logger.error(f"Error gathering results: {e}")
            raise
    
    def sync_model(self, model: torch.nn.Module) -> None:
        """Synchronize model parameters across nodes.
        
        Args:
            model: PyTorch model to synchronize
        """
        try:
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            logger.info(f"Synchronized model parameters on node {self.node_rank}")
        except Exception as e:
            logger.error(f"Error synchronizing model: {e}")
            raise
    
    def cleanup(self) -> None:
        """Clean up distributed processing group."""
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
            logger.info(f"Cleaned up distributed process group on node {self.node_rank}")
        except Exception as e:
            logger.error(f"Error cleaning up distributed group: {e}")
            raise 