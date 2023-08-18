import sys
import pytest
import warnings
warnings.filterwarnings("ignore")
from sentiment_model.processing.data_manager import load_dataset
from sentiment_model.config.core import config
from sklearn.model_selection import train_test_split

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

@pytest.fixture
def sample_input_data():
  test_data = [
    "I hardly eat chinese food for lunch!",
    "cold tea, did not enjoy it",
    "who wants to eat chinese for lunch?",
    "this food is exactly what I was looking for!!"
    ] 
  return test_data