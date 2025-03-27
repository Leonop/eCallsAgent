# Author: Zicheng Xiao
# Date: 2024-09-01
# Description: This script is used to preprocess the earnings call data.
# The data is stored in the data folder, and the preprocessed data is stored in the docword folder.

import codecs
import json
import re
import os
import string
# Fix for pydantic compatibility issue - import pydantic with specific version first
import pydantic
import logging
import sys
import importlib
import pandas as pd
import warnings
from datetime import datetime
import multiprocessing
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore")

# Import global options
from eCallsAgent.config import global_options as gl

# Try to import spaCy and transformers with error handling
try:
    import spacy
    logger.info(f"Successfully imported spaCy version: {spacy.__version__}")
except ImportError as e:
    logger.error(f"Error importing spaCy: {e}")
    spacy = None

try:
    import torch
    from transformers import BertTokenizer, BertModel
    logger.info(f"Successfully imported transformers")
except ImportError as e:
    logger.error(f"Error importing transformers: {e}")
    torch = None
    BertTokenizer = None
    BertModel = None

class NlpPreProcess(object):
    """
    Natural Language Processing class for preprocessing earnings call data.
    
    This class provides methods for text preprocessing, including:
    - Stopword removal
    - Lemmatization
    - N-gram generation
    - Punctuation and digit removal
    - Sentence filtering
    
    It supports multiple stemming algorithms:
    - Porter Stemmer: A popular stemming algorithm that removes common morphological and inflectional endings from words.
    - Snowball Stemmer: An improvement over the Porter Stemmer, also known as Porter2, which is more accurate and handles more edge cases.
    - Lancaster Stemmer: Another popular stemming algorithm that is known for its aggressive approach to stemming.
    - WordNet Lemmatizer: Uses a dictionary of known word forms to convert words to their base forms.
    """
    def __init__(self):
        super(NlpPreProcess, self).__init__()
        self.wnl = WordNetLemmatizer()  # Lemmatization
        self.ps = PorterStemmer()  # Stemming
        self.sb = SnowballStemmer('english')  # Stemming
        self.stoplist = list(set([word.strip().lower() for word in gl.stop_list]))
        
        # Try to load spacy model with error handling
        self.nlp = None
        try:
            if spacy is not None:
                self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
                logger.info("Successfully loaded spaCy model")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            logger.warning("Using fallback tokenization instead of spaCy")
        
        # Initialize BERT model and tokenizer if available
        self.model = None
        self.tokenizer = None
        try:
            if torch is not None and BertTokenizer is not None:
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                logger.info("Successfully loaded BERT tokenizer")
                # Only load model if GPU is available
                if torch.cuda.is_available():
                    self.model = BertModel.from_pretrained('bert-base-uncased')
                    self.model.eval()
                    logger.info("Successfully loaded BERT model")
        except Exception as e:
            logger.error(f"Error loading BERT model/tokenizer: {e}")
        
    def remove_stopwords_from_sentences(self, text):
        '''Split text by sentence, remove stopwords in each sentence, and rejoin sentences into one string'''
        if not text or not isinstance(text, str):
            return ""
            
        # Split text into sentences
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                # Process each sentence by removing stop words
                processed_sentences = []
                for sent in doc.sents:
                    processed_sentence = ' '.join([token.text for token in sent if token.text.lower() not in self.stoplist])
                    processed_sentences.append(processed_sentence)
                # Rejoin all processed sentences into a single string
                return ' '.join(processed_sentences)
            except Exception as e:
                logger.error(f"Error in spaCy processing: {e}")
                # Fall back to simple processing
        
        # Fallback: simple sentence splitting and stopword removal
        sentences = re.split(r'[.!?]+', text)
        processed_sentences = []
        for sentence in sentences:
            words = sentence.split()
            processed_sentence = ' '.join([word for word in words if word.lower() not in self.stoplist])
            processed_sentences.append(processed_sentence)
        return ' '.join(processed_sentences)
    def lemmatization(self, text, allowed_postags=['NOUN', 'ADJ', 'VERB']):
        '''Lemmatize and filter tokens by part-of-speech'''
        if self.nlp is None:
            # Fallback to simple word splitting if spaCy is not available
            return text.split()
            
        texts_out = []
        doc = self.nlp(text)        
        # Filter allowed POS tags and lemmatize
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out[0]  # Return flat list of lemmatized tokens

    def lemmatize_texts(self, texts):
        """Lemmatize a batch of texts for better performance."""
        if self.nlp is None:
            # Fallback to simple word splitting if spaCy is not available
            return [text.split() for text in texts]
            
        lemmatized_texts = []
        for doc in self.nlp.pipe(texts, batch_size=50, disable=["ner", "parser"]):
            lemmatized_texts.append([token.lemma_ for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB']])
        return lemmatized_texts

    def remove_stopwords(self, tokens):
        '''Remove stopwords from tokenized words'''
        # Ensure stopwords and tokens are all lowercase and stripped of spaces
        self.stoplist = {word.strip().lower() for word in self.stoplist}  # Normalize stoplist
        return [word for word in tokens if isinstance(word, str) and word.lower() not in self.stoplist]

    def remove_punct_and_digits(self, text):
        '''Remove punctuation and digits using regular expressions'''
        text = re.sub(r'[{}]'.format(string.punctuation), ' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'\d+', '', text)  # Remove digits
        return text.strip()  # Trim any leading/trailing spaces
    
    def preprocess_file(self, df, col):
        '''Preprocess the file: remove punctuation, digits, stopwords, lemmatize, and create n-grams'''
        stime = datetime.now()
        df[col] = df[col].astype(str)
        # Step 0: Final deduplication
        df = df.drop_duplicates(subset=col).reset_index(drop=True)
        print(f"Final deduplication completed in {datetime.now() - stime}")
        
        # Step 1: Remove punctuation and digits
        df[col] = df[col].progress_apply(self.remove_punct_and_digits)
        print(f"Step 1 completed in {datetime.now() - stime}")
        print(df.head())
        
        # Step 2: Tokenize into words
        if self.nlp is not None:
            df[col] = df[col].progress_apply(lambda x: [token.text for token in self.nlp(x) if not token.is_space])
        else:
            # Fallback to simple tokenization if spaCy is not available
            df[col] = df[col].progress_apply(lambda x: x.split())
        print(f"Step 2 completed in {datetime.now() - stime}")
        print(df.head())
        
        # Step 3: Remove stopwords
        df[col] = df[col].progress_apply(lambda x: self.remove_stopwords(x) if isinstance(x, list) else x)
        
        # Step 4: Apply lemmatization
        df[col] = df[col].progress_apply(lambda x: self.lemmatization(' '.join(x)) if isinstance(x, list) else x.split())
        print(f"Step 4 completed in {datetime.now() - stime}")
        print(df.head())
        
        # Step 5: Create bigrams and trigrams
        try:
            df[col] = pd.Series(self.smart_ngrams(df[col].tolist(), gl.MIN_COUNT, gl.THRESHOLD))
            print(f"Step 5 completed in {datetime.now() - stime}")
        except Exception as e:
            logger.error(f"Error in smart_ngrams: {e}")
            # Continue without n-grams if there's an error
            logger.warning("Continuing without n-grams")
        print(df.head())
        
        # Step 6: Remove stopwords from bigrams and trigrams
        # df[col] = df[col].progress_apply(lambda x: self.remove_stopwords(x) if isinstance(x, list) else x and len(str(x)) >= 2)
        print(f"Step 6 completed in {datetime.now() - stime}")
        print(df.head())
        
        # Step 7: Rejoin tokenized words into a string
        df[col] = df[col].progress_apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    
        print(f"Step 7 completed in {datetime.now() - stime}")
        print(df.head())
        print(f"Processing completed in {datetime.now() - stime}")
        return df[col]

    def remove_unnecessary_sentence(self, text):
        """去除列表裏无用句子"""
        # 輸入是原始文本str，先分句成爲list,最後把list 轉換成str
        # check if text is nan
        if not isinstance(text, str):
            return ""
        text = text.split("|||")
        # find the first index of "operator"
        f_index = 0
        try:
            f_index = text.index("Operator")
        except:
            f_index=0
        # remove the sentence before the f_index
        text = text[f_index+1:]
        # check if text is empty
        if len(text) == 0:
            return ""
        # make sure there are at 10 words in a sentnece
        text = [sentence for sentence in text if len(re.split(r'\s+', str(sentence))) >= 1]
        return text
    
    def remove_snippet(self, list_sentences):
        """
        This function removes "safe harbor" snippets from transcript sentences. Specifically, it checks the number of safe
        harbor keywords in a given snippet and a specific criteria, then removes any that matches such criteria.
        
        Arguments:
            - list_sentences: A list of sentences to search for "safe harbor" snippets.

        Return:
            - text: A list of the original transcript sentences, excluding any identified "safe harbor" snippets.
        """
        # Given safe harbor keywords to search for in each snippet
        safe_harbor_keywords = {
            'safe', 
            'harbor', 
            'forwardlooking',
            'forward-looking',
            'forward', 
            'looking',
            'actual',
            'statements', 
            'statement',
            'risk', 
            'risks', 
            'uncertainty',
            'uncertainties',
            'future',
            'events', 
            'sec',
            'results'
        }
        
        # Initialize the text list
        text = []
        
        # Iterate over the list of sentences
        list_sentences = [s for s in list_sentences if s]
        for idx, snippet in enumerate(list_sentences):
            # Split the snippet into words and count the number of safe harbor keywords it contains
            num_keywords = sum(word.lower() in safe_harbor_keywords for word in snippet.split())
            # Iterate the first half of the list of sentences
            # Remove the snippet if it has more than two safe harbor keywords or less than 2 with forward-looking or forwardlooking 
            # in its content
            if (num_keywords > 2) or (('forward-looking' in snippet.lower()) or ('forward looking' in snippet.lower())):
                return ''
            else:
                text  
        # Return the updated transcript text after removing any matching "safe harbor" snippet
        return text

# if __name__ == '__main__':
#     preprocess_file()
