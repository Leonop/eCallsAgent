import logging
import time
from threading import Thread
import torch
import psutil
import os

class GPUMemoryMonitor:
    """
    A utility class for monitoring GPU memory usage and implementing memory management strategies.
    """
    def __init__(self, device=None, warning_threshold=0.8, critical_threshold=0.9, check_interval=5):
        self.logger = logging.getLogger(__name__)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start the memory monitoring thread."""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA is not available. GPU monitoring disabled.")
            return
            
        self.monitoring = True
        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("GPU memory monitoring started")
        
    def stop_monitoring(self):
        """Stop the memory monitoring thread."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            self.logger.info("GPU memory monitoring stopped")
            
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            self._check_memory()
            time.sleep(self.check_interval)
            
    def _check_memory(self):
        """Check current memory usage and implement management strategies."""
        try:
            # Get memory statistics
            total = torch.cuda.get_device_properties(self.device).total_memory
            reserved = torch.cuda.memory_reserved(self.device)
            allocated = torch.cuda.memory_allocated(self.device)
            free = total - allocated
            
            # Calculate usage ratios
            usage_ratio = allocated / total
            reserved_ratio = reserved / total
            
            # Log current status
            self.logger.debug(
                f"GPU Memory Status - "
                f"Used: {allocated/1e9:.2f}GB ({usage_ratio*100:.1f}%), "
                f"Reserved: {reserved/1e9:.2f}GB ({reserved_ratio*100:.1f}%), "
                f"Free: {free/1e9:.2f}GB"
            )
            
            # Check system memory as well
            system_memory = psutil.virtual_memory()
            self.logger.debug(
                f"System Memory - "
                f"Used: {system_memory.used/1e9:.2f}GB ({system_memory.percent}%), "
                f"Available: {system_memory.available/1e9:.2f}GB"
            )
            
            # Implement memory management strategies
            if usage_ratio > self.critical_threshold:
                self.logger.warning("Critical GPU memory usage detected! Implementing emergency measures...")
                self._handle_critical_memory()
            elif usage_ratio > self.warning_threshold:
                self.logger.warning("High GPU memory usage detected! Implementing preventive measures...")
                self._handle_high_memory()
                
        except Exception as e:
            self.logger.error(f"Error in memory monitoring: {e}")
            
    def _handle_high_memory(self):
        """Handle high memory usage scenario."""
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Suggest garbage collection
            import gc
            gc.collect()
            
            self.logger.info("Preventive memory measures implemented")
            
        except Exception as e:
            self.logger.error(f"Error in handling high memory usage: {e}")
            
    def _handle_critical_memory(self):
        """Handle critical memory usage scenario."""
        try:
            # Emergency measures
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            # Force cache clear in PyTorch
            for obj in gc.get_objects():
                if torch.is_tensor(obj):
                    if obj.is_cuda:
                        del obj
                        
            # Log process memory info
            process = psutil.Process(os.getpid())
            self.logger.warning(f"Process memory info: {process.memory_info()}")
            
            self.logger.info("Emergency memory measures implemented")
            
        except Exception as e:
            self.logger.error(f"Error in handling critical memory usage: {e}")
            
    def get_memory_status(self):
        """
        Get current memory status.
        Returns a dictionary with memory statistics.
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
            
        try:
            total = torch.cuda.get_device_properties(self.device).total_memory
            reserved = torch.cuda.memory_reserved(self.device)
            allocated = torch.cuda.memory_allocated(self.device)
            free = total - allocated
            
            return {
                "total_memory": total,
                "reserved_memory": reserved,
                "allocated_memory": allocated,
                "free_memory": free,
                "usage_ratio": allocated / total,
                "reserved_ratio": reserved / total
            }
        except Exception as e:
            return {"error": str(e)} 