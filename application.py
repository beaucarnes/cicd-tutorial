from flask import Flask, jsonify
from recommendation.data import DataManager
from recommendation.engine import RecommendationEngine

application = Flask(__name__)  # Named 'application' for Elastic Beanstalk

# Initialize components
data_manager = DataManager('data/users.csv', 'data/products.csv')
data_manager.load_data()
engine = RecommendationEngine(data_manager)

@application.route('/health')
def health_check():
  """Health check endpoint"""
  return jsonify({"status": "healthy"})

@application.route('/recommend/<int:user_id>')
def get_recommendations(user_id):
  """Get recommendations for a user"""
  try:
      recommendations = engine.get_recommendations(user_id)
      if not recommendations:
          return jsonify({"error": "No recommendations found"}), 404
          
      # Get product details
      products = []
      for prod_id in recommendations:
          product = data_manager.products_df[
              data_manager.products_df['product_id'] == prod_id
          ].iloc[0]
          products.append({
              "id": int(prod_id),
              "name": product['name'],
              "category": product['category']
          })
          
      return jsonify({"recommendations": products})
  except Exception as e:
      return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
  application.run()