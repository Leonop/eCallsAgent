#!/usr/bin/env python3
import pandas as pd
from collections import Counter
import logging
from pathlib import Path
import sys
from tqdm import tqdm
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('word_frequency.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def clean_text(text):
    """Clean text by removing special characters and extra whitespace."""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def count_word_frequencies(file_path, chunk_size=1000):
    """
    Count word frequencies from a large CSV file using chunking.
    
    Args:
        file_path (str): Path to the CSV file
        chunk_size (int): Number of rows to process at a time
    
    Returns:
        Counter: Word frequency counter
    """
    logger.info(f"Starting word frequency analysis for {file_path}")
    
    # Initialize counter
    word_counter = Counter()
    
    try:
        # Read the CSV file in chunks
        for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size), desc="Processing chunks"):
            # Clean and process each text
            texts = chunk['componenttext'].apply(clean_text)
            
            # Count words in each text
            for text in texts:
                words = text.split()
                word_counter.update(words)
                
        logger.info(f"Completed word frequency analysis. Found {len(word_counter)} unique words.")
        return word_counter
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise

def save_frequencies(counter, output_path, top_n=1000):
    """
    Save word frequencies to a CSV file.
    
    Args:
        counter (Counter): Word frequency counter
        output_path (str): Path to save the results
        top_n (int): Number of top words to save
    """
    try:
        # Convert counter to DataFrame
        df = pd.DataFrame(counter.most_common(top_n), columns=['word', 'frequency'])
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved word frequencies to {output_path}")
        
        # Print some statistics
        total_words = sum(counter.values())
        logger.info(f"Total words processed: {total_words}")
        logger.info(f"Unique words: {len(counter)}")
        logger.info(f"Top 10 most frequent words: {dict(counter.most_common(10))}")
        
    except Exception as e:
        logger.error(f"Error saving frequencies: {e}")
        raise

if __name__ == "__main__":
    # File paths
    input_file = Path("eCallsAgent/input_data/processed/componenttext_2013_2014.csv")
    output_file = Path("eCallsAgent/output/word_frequencies_2013_2014.csv")
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Count word frequencies
    word_counter = count_word_frequencies(input_file, chunk_size=5000)
    
    # Save results
    save_frequencies(word_counter, output_file, top_n=5000)
