import numpy as np
import pandas as pd  # Add this import
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationEngine:
  def __init__(self, data_manager):
      self.data_manager = data_manager
      self.user_item_matrix = None
      self.similarity_matrix = None

  def create_user_item_matrix(self):
      """Create user-item matrix from ratings"""
      self.user_item_matrix = pd.pivot_table(
          self.data_manager.users_df,
          values='rating',
          index='user_id',
          columns='product_id',
          fill_value=0
      )
      return self.user_item_matrix

  def calculate_similarities(self):
      """Calculate user-user similarity matrix"""
      if self.user_item_matrix is None:
          self.create_user_item_matrix()
      self.similarity_matrix = cosine_similarity(self.user_item_matrix)
      return self.similarity_matrix

  def get_recommendations(self, user_id, n=5):
      """Get top N recommendations for user"""
      if self.similarity_matrix is None:
          self.calculate_similarities()

      try:
          user_idx = list(self.user_item_matrix.index).index(user_id)
      except ValueError:
          return []

      # Find similar users
      similar_scores = self.similarity_matrix[user_idx]
      similar_users = np.argsort(similar_scores)[-6:]  # Get 6 users (including self)
      similar_users = similar_users[:-1]  # Remove self

      # Get rated products
      user_rated = set(self.data_manager.get_user_ratings(user_id)['product_id'])
      
      # Collect recommendations
      recommendations = []
      for idx in similar_users:
          similar_user_id = self.user_item_matrix.index[idx]
          similar_user_ratings = self.data_manager.get_user_ratings(similar_user_id)
          # Get highly rated products (rating > 3)
          recommended = similar_user_ratings[
              similar_user_ratings['rating'] > 3
          ]['product_id'].tolist()
          recommendations.extend([p for p in recommended if p not in user_rated])

      # Return top N unique recommendations
      return list(dict.fromkeys(recommendations))[:n]