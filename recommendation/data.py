import pandas as pd

class DataManager:
  def __init__(self, users_file, products_file):
      """Initialize with paths to data files"""
      self.users_file = users_file
      self.products_file = products_file
      self.users_df = None
      self.products_df = None

  def load_data(self):
      """Load user and product data from CSV files"""
      self.users_df = pd.read_csv(self.users_file)
      self.products_df = pd.read_csv(self.products_file)
      return self.users_df, self.products_df

  def get_user_ratings(self, user_id):
      """Get all ratings for a specific user"""
      return self.users_df[self.users_df['user_id'] == user_id]