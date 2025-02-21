import os
import sys
import logging
import traceback

from eCallsAgent.utils.cuda_setup import setup_cuda
from eCallsAgent.utils.logging_utils import setup_logging, log_system_info, log_gpu_info
from eCallsAgent.core.data_handler import DataHandler
from eCallsAgent.core.embedding_generator import EmbeddingGenerator
from eCallsAgent.core.topic_modeler import TopicModeler
import eCallsAgent.config.global_options as gl

# Configure logging
log_file = os.path.join(gl.log_folder, 'bertopic_processing.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ... rest of the code ... 