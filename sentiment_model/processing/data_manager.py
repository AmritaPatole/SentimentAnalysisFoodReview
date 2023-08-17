import json
import sys
import pdb
from pathlib import Path
import typing as t
from pathlib import Path

import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sentiment_model import __version__ as _version
from sentiment_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR
from sentiment_model.config.core import TRAINED_TOKENIZER_DIR, config

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

##  Pre-Pipeline Preparation
# Extract time
def ExtractTime(dataframe: pd.DataFrame, time_var: str) -> pd.DataFrame:
    df = dataframe.copy()
    df[time_var] = pd.to_datetime(df[time_var], format='%Y-%m-%d')
    return df
  
# Add 'Sentiment' column
def AddSentiment(dataframe: pd.DataFrame, score_var: str, sentiment_var: str) -> pd.DataFrame:
  df = dataframe.copy()
  df[sentiment_var] = df[score_var].apply(lambda x:'Positive' if x > 2 else 'Negative')
  return df

def convert_sentiment_to_numerical(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df[config.model_config.sentiment_var] = df[config.model_config.sentiment_var].apply(lambda x: 1 if x=="positive" else 0)
    return df

def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    # Update timestamp in yy-mm-dd format
    data_frame = ExtractTime(dataframe=data_frame, time_var=config.model_config.time_var)
    
    # Add new column 'Sentiment' based on 'Score'
    data_frame = AddSentiment(dataframe = data_frame, score_var = config.model_config.score_var,
                              sentiment_var= 'Sentiment')
    
    # Drop unnecessary fields
    for field in config.model_config.unused_fields:
        if field in data_frame.columns:
            data_frame.drop(labels = field, axis=1, inplace=True)    

    # Drop duplicates
    data_frame.drop_duplicates(subset=[config.model_config.sentiment_var, config.model_config.text_var])
    
    # convert 'Sentiment' column to numerical
    data_frame = convert_sentiment_to_numerical(data_frame)
    
    return data_frame

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))    
    transformed = pre_pipeline_preparation(data_frame=dataframe)
    return transformed

def remove_old_model(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old models.
    This is to ensure there is a simple one-to-one mapping between the package version and 
    the model version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
            
def get_model_filepath():
    # Prepare versioned save file name
    save_file_name = f"{config.app_config.model_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_model(files_to_keep = [save_file_name])
    return str(save_path)


def load_model(*, file_name: str) -> keras.models.Model:
    """Load a persisted model."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = keras.models.load_model(filepath = file_path)
    return trained_model
            
def save_tokenizer(tokenizer, filename):
    json_string = tokenizer.to_json()
    filepath = TRAINED_TOKENIZER_DIR / filename
    # remove existing file
    if Path(filepath).is_file():
        Path(filepath).unlink()
    # write to a file
    with open(str(filepath), 'w') as f:
        json.dump(json_string, f)
    print(f"Creating new tokenizer file at path: {filepath}")
    
def load_tokenizer(filename):
    filepath = Path(f"{TRAINED_TOKENIZER_DIR}/{filename}")
    if filepath.is_file():
        print(f"Found saved tokenizer at {filepath}")
        with open(filepath) as f:
            json_string = json.load(f)
            return tokenizer_from_json(json_string)
    return None
    
def create_tokenizer(X_train):
    # create new tokenizer
    tokenizer = Tokenizer(num_words=config.model_config.num_tokens)
    tokenizer.fit_on_texts(X_train)
    return tokenizer

    