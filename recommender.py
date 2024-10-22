import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationSystem:
  def __init__(self, user_data, product_data):
      self.user_data = pd.read_csv(user_data)
      self.product_data = pd.read_csv(product_data)
      
  def preprocess_data(self):
      # Merge user and product data
      self.data = pd.merge(self.user_data, self.product_data, on='product_id')
      
      # Create user-item matrix
      self.user_item_matrix = self.data.pivot_table(
          index='user_id', columns='product_id', values='rating'
      ).fillna(0)
      
  def generate_recommendations(self, user_id, n=5):
      if user_id not in self.user_item_matrix.index:
          return []
          
      # Get user's ratings
      user_ratings = self.user_item_matrix.loc[user_id]
      
      # Find products the user hasn't rated yet
      unrated_products = user_ratings[user_ratings == 0].index
      
      # Calculate similarity between users
      similarity_scores = cosine_similarity(
          self.user_item_matrix.loc[[user_id]], 
          self.user_item_matrix
      )[0]
      
      # Get top similar users (excluding the user themselves)
      similar_users_idx = similarity_scores.argsort()[::-1][1:6]  # top 5 similar users
      similar_users = self.user_item_matrix.index[similar_users_idx]
      
      # Calculate predicted ratings for unrated products
      predicted_ratings = {}
      for product_id in unrated_products:
          # Get ratings for this product from similar users
          product_ratings = []
          for similar_user in similar_users:
              rating = self.user_item_matrix.loc[similar_user, product_id]
              if rating > 0:  # only consider actual ratings (not 0)
                  similarity = similarity_scores[self.user_item_matrix.index.get_loc(similar_user)]
                  product_ratings.append((rating, similarity))
          
          # Calculate weighted average rating if we have ratings
          if product_ratings:
              weighted_sum = sum(rating * similarity for rating, similarity in product_ratings)
              sum_similarities = sum(similarity for _, similarity in product_ratings)
              predicted_ratings[product_id] = weighted_sum / sum_similarities
      
      # Sort products by predicted rating and return top N
      recommended_products = sorted(
          predicted_ratings.items(), 
          key=lambda x: x[1], 
          reverse=True
      )[:n]
      
      return [product_id for product_id, _ in recommended_products]

if __name__ == "__main__":
  rs = RecommendationSystem('user_data.csv', 'product_data.csv')
  rs.preprocess_data()
  recommendations = rs.generate_recommendations(user_id=1, n=5)
  print(f"Recommended products for user 1: {recommendations}")
  
  # Print additional information for clarity
  print("\nRecommended products details:")
  for prod_id in recommendations:
      product_info = rs.product_data[rs.product_data['product_id'] == prod_id].iloc[0]
      print(f"Product ID: {prod_id}, Name: {product_info['product_name']}")