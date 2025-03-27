"""
eCallsAgent - A package for analyzing earnings call transcripts using LLM and topic modeling
"""

__version__ = '0.1.0'

from .core.data_handler import DataHandler
from .core.embedding_generator import EmbeddingGenerator
from .core.model_eval import ModelEvaluator
from .core.topic_modeler import TopicModeler
from .core.visualization import TopicVis
from .utils.cuda_setup import setup_cuda
from .utils.logging_utils import setup_logging

__all__ = [
    'DataHandler',
    'EmbeddingGenerator',
    'ModelEvaluator',
    'TopicModeler',
    'TopicVis',
    'setup_cuda',
    'setup_logging'
]