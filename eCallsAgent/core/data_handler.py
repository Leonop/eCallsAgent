"""
Module for handling data loading and preprocessing operations.
"""

import os
import logging
import pandas as pd
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm
from eCallsAgent.core.preprocess_earningscall import NlpPreProcess
from eCallsAgent.config import global_options as gl
import multiprocessing as mp

logger = logging.getLogger(__name__)

class DataHandler:
    """Handles data loading and preprocessing operations."""
    def __init__(self, file_path: str, year_start: int, year_end: int):
        self.file_path = file_path
        self.year_start = year_start
        self.year_end = year_end
        self.nlp_processor = NlpPreProcess()

    @staticmethod
    def _process_chunk(chunk: str) -> list:
        """Process a chunk of text into non-empty stripped lines."""
        try:
            return [line.strip() for line in chunk.splitlines() if line.strip()]
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            return []

    def load_doc_parallel(self, docs_path: str, chunk_size: int = 1024*1024) -> list:
        """Load documents from a file iteratively and process chunks in parallel."""
        docs = []
        logger.info(f"Loading documents from {docs_path}")

        def read_chunks(fp, size):
            while True:
                chunk = fp.read(size)
                if not chunk:
                    break
                yield chunk

        # Use ProcessPoolExecutor for parallel processing of chunks
        with open(docs_path, 'r', encoding='utf-8') as f, ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            futures = {executor.submit(DataHandler._process_chunk, chunk): chunk 
                       for chunk in read_chunks(f, chunk_size)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading documents"):
                docs.extend(future.result())
        logger.info(f"Loaded {len(docs)} documents")
        return docs

    def load_data(self) -> pd.DataFrame:
        """Load and filter CSV data by year."""
        logger.info(f"Loading data for years {self.year_start}-{self.year_end}")
        try:
            df_header = pd.read_csv(self.file_path, nrows=0)
            expected_cols = len(df_header.columns)
            logger.info(f"Expected columns: {expected_cols}\nColumns: {df_header.columns.tolist()}")

            chunks = pd.read_csv(
                self.file_path,
                chunksize=gl.CHUNK_SIZE,
                quotechar='"',
                doublequote=True,
                encoding='utf-8',
                engine='c',
                on_bad_lines='warn',
                delimiter=',',
                quoting=1
            )
            meta = []
            total_rows = 0
            estimated_rows = os.path.getsize(self.file_path) // 500  # rough estimate

            with tqdm(total=estimated_rows, desc="Loading data", ncols=100, colour="green") as pbar:
                for chunk in chunks:
                    if len(chunk.columns) != expected_cols:
                        logger.warning(f"Found {len(chunk.columns)} columns, expected {expected_cols}")
                        continue
                    chunk['year'] = pd.to_datetime(chunk['mostimportantdateutc'], errors='coerce').dt.year
                    filtered_chunk = chunk[(chunk['year'] >= self.year_start) & (chunk['year'] <= self.year_end)]
                    if not filtered_chunk.empty:
                        meta.append(filtered_chunk)
                        total_rows += len(filtered_chunk)
                        if total_rows % (gl.CHUNK_SIZE * 10) == 0:
                            logger.info(f"Processed {total_rows} rows")
                    pbar.update(len(chunk))
            df_meta = pd.concat(meta, ignore_index=True)
            logger.info(f"Final dataset size: {len(df_meta)} rows")
            return df_meta

        except Exception as e:
            logger.error(f"Error in load_data: {e}")
            logger.error(traceback.format_exc())
            raise

    def preprocess_text(self, data: pd.DataFrame) -> list:
        """Preprocess text data efficiently."""
        try:
            # Set up multiprocessing with explicit start method
            ctx = mp.get_context('spawn')  # Use 'spawn' instead of 'fork'
            
            # Calculate chunk size
            n_cores = mp.cpu_count()
            chunk_size = max(1, len(data) // (n_cores * 4))
            
            # Process in parallel
            with ctx.Pool(n_cores) as pool:
                docs = pool.map(
                    self._preprocess_single_doc,
                    data[gl.TEXT_COLUMN].values,
                    chunksize=chunk_size
                )
            
            return docs
        
        except Exception as e:
            logger.error(f"Error in preprocess_text: {e}")
            raise

    def _preprocess_single_doc(self, text: str) -> str:
        """Preprocess a single document."""
        return self.nlp_processor.preprocess_file(pd.DataFrame([{'text': text}]), 'text')[0] 