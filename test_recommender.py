import pytest
from recommender import RecommendationSystem
import pandas as pd

def test_recommendation_system():
  # Initialize the system
  rs = RecommendationSystem('test_user_data.csv', 'test_product_data.csv')
  rs.preprocess_data()
  
  # Test recommendations for user 1
  recommendations = rs.generate_recommendations(user_id=1, n=5)
  
  # Basic checks
  assert isinstance(recommendations, list), "Recommendations should be a list"
  assert len(recommendations) <= 5, "Should not return more than n recommendations"
  assert all(isinstance(x, int) for x in recommendations), "All recommendations should be product IDs (integers)"
  
  # Check that recommended products are valid products from our dataset
  valid_products = set(pd.read_csv('test_product_data.csv')['product_id'])
  assert all(r in valid_products for r in recommendations), "All recommendations should be valid product IDs"
  
  # Check that we don't recommend products the user has already rated
  user_rated_products = set(pd.read_csv('test_user_data.csv')[
      pd.read_csv('test_user_data.csv')['user_id'] == 1
  ]['product_id'])
  assert not any(r in user_rated_products for r in recommendations), "Should not recommend products user has already rated"
  
  # Test with invalid user
  invalid_recommendations = rs.generate_recommendations(user_id=999, n=5)
  assert len(invalid_recommendations) == 0, "Should return empty list for invalid user"

def test_data_loading():
  rs = RecommendationSystem('test_user_data.csv', 'test_product_data.csv')
  assert isinstance(rs.user_data, pd.DataFrame), "User data should be loaded as DataFrame"
  assert isinstance(rs.product_data, pd.DataFrame), "Product data should be loaded as DataFrame"
  
def test_preprocessing():
  rs = RecommendationSystem('test_user_data.csv', 'test_product_data.csv')
  rs.preprocess_data()
  assert hasattr(rs, 'user_item_matrix'), "Preprocessing should create user_item_matrix"
  assert isinstance(rs.user_item_matrix, pd.DataFrame), "user_item_matrix should be a DataFrame" 