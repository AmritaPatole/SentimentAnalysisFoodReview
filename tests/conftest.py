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
    "Thank you! I was craving for this ice cream for so long!!",
    "I hate to see how inflation has increases for small food items?",
    "small portions, perfect for light dinners",
    "will buy this food again."
    ] 
  return test_data