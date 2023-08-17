
import pdb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers

# removing the html strips
def strip_html(text):
    # BeautifulSoup is a useful library for extracting data from HTML and XML documents
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# removing punctuations
def remove_punctuations(text):
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern,'',text)

    # Single character removal
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)

    # Removing multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text

def get_stopwords():
    # setting english stopwords
    stopword_list = nltk.corpus.stopwords.words('english')
    #print(stopword_list)
    # Exclude 'not' and its other forms from the stopwords list

    updated_stopword_list = []
    for word in stopword_list:
        if word=='not' or word.endswith("n't"):
            pass
        else:
            updated_stopword_list.append(word)
    
    return updated_stopword_list

# removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    updated_stopword_list = get_stopwords()
    # splitting strings into tokens (list of words)
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        # filtering out the stop words
        filtered_tokens = [token for token in tokens if token not in updated_stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in updated_stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def clean_dataset(text:str) -> str:
    cleaned_text = strip_html(text)
    cleaned_text = remove_punctuations(cleaned_text)
    cleaned_text = remove_stopwords(cleaned_text)
    return cleaned_text