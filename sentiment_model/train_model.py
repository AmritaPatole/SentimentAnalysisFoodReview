import sys
from pathlib import Path
import json
import nltk
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split


file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_squared_error, r2_score

from sentiment_model.config.core import config
from sentiment_model.model import create_model
from sentiment_model.processing.data_manager import load_dataset, load_tokenizer, callbacks_and_save_model, create_tokenizer, save_tokenizer
from sentiment_model.processing.feature import clean_dataset

def run_training() -> None:
    # read dataset
    data = load_dataset(file_name = config.app_config.training_data_file)
    print("Read data successfully")
    
    # clean dataset
    data = clean_dataset(text=data[config.model_config.text_var].values)
    print("Preprocessing complete")
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data[config.model_config.text_var].values, 
                     data[config.model_config.sentiment_var].values,
                     test_size=config.model_config.test_size,
                     random_state=config.model_config.random_state)
    print("Divide test and train dataset")
    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(data[config.model_config.text_var].values, 
                     data[config.model_config.sentiment_var].values,
                     test_size=config.model_config.validation_size,
                     random_state=config.model_config.random_state)
    print("Divide train and validation dataset")
    print("load tokenizer")
    # load tokenizer
    tokenizer = load_tokenizer(filename = config.app_config.tokenizer_json_file)
    if tokenizer == None:
        print("Create tokenizer")
        tokenizer = create_tokenizer(X_train = X_train)
        save_tokenizer(tokenizer=tokenizer, filename=config.app_config.tokenizer_json_file)
    
    X_train_tok = tokenizer.texts_to_sequences(X_train)
    X_val_tok = tokenizer.texts_to_sequences(X_val)
    X_test_tok = tokenizer.texts_to_sequences(X_test)
            
    config.model_config.vocab_size_var = len(tokenizer.word_index) + 1
    maxlen = config.model_config.max_len
            
    X_train_pad = pad_sequences(X_train_tok, padding='post', maxlen=maxlen, truncating='post')
    X_test_pad = pad_sequences(X_test_tok, padding='post', maxlen=maxlen, truncating='post')
    X_val_pad = pad_sequences(X_val_tok, padding='post', maxlen=maxlen, truncating='post')
    
    # create a model
    model = create_model()
    
    history = model.fit(X_train_pad, y_train, batch_size=config.model_config.batch_size, epochs=config.model_config.epochs, verbose=1,
                        validation_data = (X_val_pad,y_val), callbacks = callbacks_and_save_model)
    
    print('Testing...')
    y_test = np.array(y_test)
    score, acc = model.evaluate(X_test_pad, y_test, batch_size=config.model_config.batch_size)
    
    print('Test score:', score)
    print('Test accuracy:', acc)
    print("Accuracy: {0:.2%}".format(acc))

    
if __name__ == "__main__":
    run_training()