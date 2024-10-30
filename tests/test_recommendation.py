import pytest
import pandas as pd
from recommendation.data import DataManager
from recommendation.engine import RecommendationEngine
import os

# Get the directory containing the test file
TEST_DIR = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def data_manager():
  """Create a DataManager instance with test data"""
  users_file = os.path.join(TEST_DIR, 'test_data', 'users.csv')
  products_file = os.path.join(TEST_DIR, 'test_data', 'products.csv')
  dm = DataManager(users_file, products_file)
  dm.load_data()
  return dm

def test_data_loading(data_manager):
  """Test that data is loaded correctly"""
  assert data_manager.users_df is not None
  assert data_manager.products_df is not None
  assert len(data_manager.users_df) > 0
  assert len(data_manager.products_df) > 0

def test_recommendations(data_manager):
  """Test recommendation generation"""
  engine = RecommendationEngine(data_manager)
  recommendations = engine.get_recommendations(user_id=1)
  
  assert isinstance(recommendations, list)
  assert len(recommendations) <= 5
  assert all(isinstance(x, int) for x in recommendations)

def test_invalid_user(data_manager):
  """Test handling of invalid user ID"""
  engine = RecommendationEngine(data_manager)
  recommendations = engine.get_recommendations(user_id=999)
  assert recommendations == []

def test_user_ratings(data_manager):
  """Test getting user ratings"""
  user_ratings = data_manager.get_user_ratings(1)
  assert len(user_ratings) > 0
  assert 'rating' in user_ratings.columns
  assert 'product_id' in user_ratings.columns