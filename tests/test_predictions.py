"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from sentiment_model import __version__ as _version
from sentiment_model.config.core import config
from sentiment_model.predict import make_prediction

def test_make_prediction(sample_input_data):
    # Given
    for text in sample_input_data:
        predicted_sentiment, sentiment_probability = make_prediction(text)
        print("Predicted Sentiment:", predicted_sentiment)
        print("Sentiment Probability:", sentiment_probability)


def test_precision(sample_input_data):   
    for text in sample_input_data:
        predicted_sentiment, sentiment_probability = make_prediction(text)
        print("Predicted Sentiment:", predicted_sentiment)
        print("Sentiment Probability:", sentiment_probability)
        
    assert sentiment_probability > 0.8
    