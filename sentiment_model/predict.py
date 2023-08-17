import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from sentiment_model import __version__ as _version
from sentiment_model.config.core import config
from sentiment_model.processing.data_manager import load_model, load_tokenizer
from sentiment_model.processing.feature import clean_dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences

model_file_name = f"{config.app_config.model_save_file}{_version}.pkl"
food_sentiment_model = load_model(file_name = model_file_name)

def make_prediction(input_text):
    # Preprocess the input text
    cleaned_input = clean_dataset(input_text)
    
    # Tokenize and pad the preprocessed text
    tokenizer = load_tokenizer(filename = config.app_config.tokenizer_json_file)
    assert(tokenizer != None)
    
    cleaned_input_tok = tokenizer.texts_to_sequences(cleaned_input)
    vocab_size = config.model_config.vocab_size_var
    maxlen = config.model_config.max_len
            
    cleaned_input_pad = pad_sequences(cleaned_input_tok, padding='post', maxlen=maxlen, truncating='post')
    
    # Predict sentiment
    sentiment_probability = food_sentiment_model.predict(cleaned_input_pad)[0][0]
    predicted_sentiment = 'positive' if sentiment_probability > 0.5 else 'negative'

    return predicted_sentiment, sentiment_probability

if __name__ == "__main__":
    test_data = [
         "I hardly eat chinese food for lunch!",
         "cold tea, did not enjoy it",
         "who wants to eat chinese for lunch?",
         "this food is exactly what I was looking for!!"
     ] 
    
    for data in test_data:
        user_input = data
        predicted_sentiment, sentiment_probability = make_prediction(user_input)
        print("Predicted Sentiment:", predicted_sentiment)
        print("Sentiment Probability:", sentiment_probability)