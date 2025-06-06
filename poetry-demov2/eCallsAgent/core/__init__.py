"""
Core components for narrativesLLM package.
"""

# Set Numba CUDA compatibility environment variable at the module level
import os


from .data_handler import DataHandler
from .embedding_generator import EmbeddingGenerator
from .topic_modeler import TopicModeler
from .visualization import TopicVis
from eCallsAgent.config import global_options as gl

__all__ = [
    'DataHandler', 
    'EmbeddingGenerator', 
    'TopicModeler',
    'TopicVis',
    'gl'
]

