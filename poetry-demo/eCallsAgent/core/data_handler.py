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
import numpy as np     
tqdm.pandas()

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

    def preprocess_text_parallel(self, data: pd.DataFrame) -> list:
        """Preprocess text data efficiently."""
        try:
            # Set up multiprocessing with explicit start method
            ctx = mp.get_context('spawn')  # Use 'spawn' instead of 'fork'
            
            # Calculate optimal chunk size based on data size
            n_cores = mp.cpu_count()
            chunk_size = max(1, len(data) // (n_cores * 4))
            
            # Process in parallel with progress bar
            with ctx.Pool(n_cores) as pool:
                docs = list(tqdm(
                    pool.imap(
                        self._preprocess_single_doc,
                        data[gl.TEXT_COLUMN].values,
                        chunksize=chunk_size
                    ),
                    total=len(data),
                    desc="Preprocessing documents in parallel"
                ))
            return docs
        
        except Exception as e:
            logger.error(f"Error in preprocess_text: {e}")
            raise

    def _preprocess_text(self, df: pd.DataFrame, col: str) -> str:
        """Preprocess a single document."""
        try:
            return self.nlp_processor.preprocess_file(df, col).tolist()
        except Exception as e:
            logger.error(f"Error preprocessing document: {e}")
            return ""

    def _create_topic_probabilities_csv(self, df_meta: pd.DataFrame, docs: list, embeddings: np.ndarray, topic_modeler, output_path):
        """
        Create a CSV file that combines document identifiers with topic probabilities.
        
        Args:
            csv_path: Path to the raw transcript CSV
            topic_modeler: Your TopicModeler instance containing rep_topics and rep_probs
            output_path: Where to save the resulting CSV
        """
        
        # Filter for years 2011-2014
        filtered_df = df_meta[(df_meta['year'] >= self.year_start) & (df_meta['year'] <= self.year_end)].copy()
        logger.info(f"Filtered to {len(filtered_df)} documents from {self.year_start}-{self.year_end}")
        
        # Create a unique identifier (using transcriptid)
        # You can adjust this if you need a different identifier
        identifiers = filtered_df['transcriptid'].values
        
        # Check if lengths match
        if len(identifiers) != len(topic_modeler.rep_probs):
            logger.info(f"WARNING: Length mismatch! Identifiers: {len(identifiers)}, Probabilities: {len(topic_modeler.rep_probs)}")
            return False
        
        logger.info(f"Lengths match! Creating CSV with {len(identifiers)} rows")
        
        topics, probs = topic_modeler._map_documents(docs, embeddings)

        # Create a DataFrame with identifiers and probabilities
        # If rep_probs is a 2D array (probabilities for each topic)
        if len(probs) > 1:
            # Create column names for each topic probability
            topic_cols = [f'topic_{i}_embedding' for i in range(probs.shape[1])]
            
            # Create DataFrame
            result_df = pd.DataFrame(probs, columns=topic_cols)
            result_df.insert(0, 'transcriptid', identifiers)
            result_df.insert(1, 'assigned_topic', topics)
            result_df.insert(2, 'topic_probability', probs)
        else:
            # If rep_probs is 1D (just the confidence for the assigned topic)
            result_df = pd.DataFrame({
                'transcriptid': identifiers,
                'assigned_topic': topics,
                'topic_probability': probs
            })
        
        # Save to CSV
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved topic probabilities to {output_path}")
        
        return True

    def process_chunk(self, chunk_df, groupby_cols, rows_to_keep):
        chunk_result = []
        chunk_skipped = 0
        for name, group in chunk_df.groupby(groupby_cols):
            if group['transcriptcomponenttypename'].iloc[0] in rows_to_keep:
                first_row = group.iloc[0].copy()
                first_row['componenttext'] = group['componenttext'].iloc[-1]
                first_row['transcriptid'] = group['transcriptid'].iloc[0]
                chunk_result.append(first_row)
            else:
                chunk_skipped += 1
        return chunk_result, chunk_skipped
        
    def process_dup_earnings_calls(self, df_input: pd.DataFrame):
        # Load the data
        df = df_input
        logger.info(f"Successfully loaded data with {len(df)} rows")  
        df['mostimportantdateutc'] = pd.to_datetime(df['mostimportantdateutc'])
        # 1. Add quarter column based on mostimportantdateutc
        df['quarter'] = df['mostimportantdateutc'].dt.quarter
        df['year'] = df['mostimportantdateutc'].dt.year
        df['componentorder'] = df['componentorder'].astype(int)

        # Sort the dataframe
        df = df.sort_values(['transcriptid', 'companyid', 'mostimportantdateutc', 'transcriptcomponenttypename', 'componentorder'], ascending=True)

        # Define the grouping columns
        groupby_cols = ['companyid', 'year', 'quarter', 'transcriptcomponenttypename', 'componentorder']
        rows_to_keep = ['Presenter Speech', 'Question', 'Answer']
        
        # Calculate optimal chunk size - aim for ~100 chunks
        n_groups = df.groupby(groupby_cols).ngroups
        n_cores = min(mp.cpu_count(), 16)  # Limit to 16 cores max
        chunk_size = max(1000, n_groups // (n_cores * 4))  # Ensure reasonable chunk size
        
        # Create chunks of the dataframe
        unique_transcripts = df['transcriptid'].unique()
        transcript_chunks = np.array_split(unique_transcripts, n_cores * 4)
        
        # Process chunks in parallel
        result = []
        count_skipped = 0
        
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = []
            for transcript_chunk in transcript_chunks:
                chunk_df = df[df['transcriptid'].isin(transcript_chunk)]
                futures.append(executor.submit(self.process_chunk, chunk_df, groupby_cols, rows_to_keep))
            
            for future in as_completed(futures):
                chunk_result, chunk_skipped = future.result()
                result.extend(chunk_result)
                count_skipped += chunk_skipped
        
        logger.info(f"Skipped {count_skipped} rows for not in rows_to_keep")
        df_unique_calls = pd.DataFrame(result)
        logger.info(f"There are {len(df_unique_calls)} unique calls")
        # drop duplicates text
        df_unique_calls = df_unique_calls.drop_duplicates(subset='componenttext', keep='first')
        logger.info(f"There are {len(df_unique_calls)} unique calls after dropping duplicates")
        self.save_data(df_unique_calls, 'transcriptid', 'componenttext', gl.YEAR_START, gl.YEAR_END)
        return df_unique_calls[gl.TEXT_COLUMN].astype(str).tolist()

    def save_data(self, df: pd.DataFrame, id_name: str, column_name: str, start_year: int, end_year: int):
        # Create output directory if it doesn't exist
        os.makedirs(gl.input_folder, exist_ok=True)
        
        # Save as proper CSV with both columns
        output_file = os.path.join(gl.input_folder, 'processed', f'{column_name}_{start_year}_{end_year}.csv')
        
        # Create a new DataFrame with just the two columns we want
        output_df = df[[id_name, column_name]].copy()
        
        # Save to CSV with proper header
        output_df.to_csv(output_file, index=False)
        logger.info(f"Saved data to {output_file} with {len(output_df)} rows")