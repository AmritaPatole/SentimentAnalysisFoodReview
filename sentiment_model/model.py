import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers

from sentiment_model.config.core import config

def create_model():
    print('Build model...')
    
    model = Sequential()
    model.add(Embedding(input_dim = config.model_config.vocab_size_var, output_dim = config.model_config.embedding_dim, input_length= config.model_config.max_len))
    model.add(LSTM(units=config.model_config.lstm_units, dropout=config.model_config.lstm_dropouts, recurrent_dropout=config.model_config.lstm_recurrent_dropouts))
    model.add(Dense(1, activation=config.model_config.activation))

    # Try using different optimizers and different optimizer configs
    model.compile(loss=config.model_config.loss, optimizer=config.model_config.optimizer, metrics=[config.model_config.metrics])

    print('Summary of the built model...')
    print(model.summary())
    
    return model