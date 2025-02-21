# Author: Zicheng Xiao
# Date: 2024-09-01
# Description: This script is used to preprocess the earnings call data.
# The data is stored in the data folder, and the preprocessed data is stored in the docword folder.


import codecs
import json
import re
import os
import string
# import nltk
import spacy
import torch
from transformers import BertTokenizer, BertModel
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet
from datetime import datetime
import multiprocessing
import pandas as pd
# ignore the warning
import warnings
warnings.filterwarnings("ignore")
import importlib,sys
importlib.reload(sys)
# sys.setdefaultencoding('utf-8')
path = sys.path[0]
import nltk
import datetime as dt
from eCallsAgent.config import global_options as gl
import warnings
import gensim
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import spacy
from tqdm import tqdm
tqdm.pandas()
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger')
warnings.filterwarnings("ignore")

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force CPU usage for BERT
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
device = torch.device('cpu')
logger.info(f"Using device: {device}")

# Initialize tokenizer and model on CPU
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()  # Set to evaluation mode
    logger.info("BERT model loaded successfully on CPU")
except Exception as e:
    logger.error(f"Error loading BERT model: {e}")
    raise

class NlpPreProcess(object):
    """preprocess the html of gdelt-data
    arg: str
        filepath of the stoplist-file
    function:
        preprocess_folder(source_folder, dest_folder): preprocess all files in the source-folder, and save the results to dest-folder
        preprocess_file(doc): preprocess a doc, delete the punctuation,digit,meanningless word
        generate_dict_from_file(filename): generator to parse dict from json file
    # >>>nlp_preprocess = NlpPreProcess('')
    """
    """
    Snowball Stemmer: The Snowball Stemmer, also known as the Porter2 stemmer, is an extension of the PorterStemmer algorithm that supports several languages and has been shown to produce better results in some cases.

    Lancaster Stemmer: The Lancaster Stemmer is another popular stemming algorithm that is known for its aggressive approach to stemming, which can result in very short stems but may also result in more aggressive stemming.

    WordNet Lemmatizer: While not a stemming algorithm per se, the WordNet Lemmatizer is a powerful tool for reducing words to their base forms. Unlike stemming algorithms, which simply remove affixes from words, the WordNet Lemmatizer uses a dictionary of known word forms to convert words to their base forms, which can be more accurate in some contexts.

    RSLP Stemmer: The RSLP Stemmer is a stemming algorithm that was specifically designed for the Portuguese language, but has also been applied to other languages with some success.

    Lovins Stemmer: The Lovins Stemmer is a stemming algorithm that was designed to be less aggressive than the PorterStemmer, with the aim of producing stems that are more readable and closer to the original words. However, it may also result in less stemming in some cases.
        """
    def __init__(self):
        super(NlpPreProcess, self).__init__()
        self.wnl = WordNetLemmatizer() # 词形还原
        self.ps = PorterStemmer() # 词干提取
        self.sb = SnowballStemmer('english') # 词干提取
        self.stoplist = list(set([word.strip().lower() for word in gl.stop_list]))
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # Disable NER and dependency parser for speed
        # Store model reference
        self.model = model
        self.tokenizer = tokenizer
        
    def remove_stopwords_from_sentences(self, text):
        '''Split text by sentence, remove stopwords in each sentence, and rejoin sentences into one string'''
        # Split text into sentences
        sentences = self.nlp(text)
        # Process each sentence by removing stop words
        processed_sentences = []
        for sentence in sentences:
            processed_sentence = ' '.join([word for word in sentence.split() if word.lower() not in self.stoplist])
            processed_sentences.append(processed_sentence)
        # Rejoin all processed sentences into a single string
        return ' '.join(processed_sentences)
    
    
    def smart_ngrams(self, docs, min_count, threshold):
        """
        Create meaningful bigrams and trigrams from the list of tokenized documents.
        :param docs: List of tokenized documents
        :param min_count: Minimum number of times a bigram/trigram should occur to be considered.
        :param threshold: Higher threshold means fewer phrases; controls the tendency to form phrases.
        :return: List of tokenized documents with bigrams and trigrams
        """
        # Train the bigram model
        bigram_phrases = Phrases(docs, min_count=min_count, threshold=threshold,  delimiter='_')
        bigram = Phraser(bigram_phrases)

        # Train the trigram model on the bigram-transformed documents
        trigram_phrases = Phrases(docs, min_count=min_count, threshold=threshold, delimiter='_')
        trigram = Phraser(trigram_phrases)

        # Transform documents into bigrams and trigrams
        bigram_trigram_docs = [
            bigram[doc] + [token for token in trigram[bigram[doc]] if '_' in token] for doc in docs
        ]
        return bigram_trigram_docs
    
    def lemmatization(self, text, allowed_postags=['NOUN', 'ADJ', 'VERB']):
        '''Lemmatize and filter tokens by part-of-speech'''
        texts_out = []
        doc = self.nlp(text)        
        # Filter allowed POS tags and lemmatize
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out[0]  # Return flat list of lemmatized tokens

    def lemmatize_texts(self, texts):
        """Lemmatize a batch of texts for better performance."""
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
        # df[col] = df[col].progress_apply(nltk.word_tokenize)
        df[col] = df[col].progress_apply(lambda x: [token.text for token in self.nlp(x) if not token.is_space])
        print(f"Step 2 completed in {datetime.now() - stime}")
        print(df.head())
        # Step 3: Remove stopwords
        df[col] = df[col].progress_apply(lambda x: self.remove_stopwords(x) if isinstance(x, list) else x)
        # print(f"Step 3 completed in {datetime.now() - stime}")   
        # print(df.loc[1, col]) 
        # Step 4: Apply lemmatization
        df[col] = df[col].progress_apply(lambda x: self.lemmatization(' '.join(x)))
        print(f"Step 4 completed in {datetime.now() - stime}")
        print(df.head())
        # Step 5: Create bigrams and trigrams
        df[col] = pd.Series(self.smart_ngrams(df[col].tolist(), gl.MIN_COUNT, gl.THRESHOLD))
        print(f"Step 5 completed in {datetime.now() - stime}")
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
