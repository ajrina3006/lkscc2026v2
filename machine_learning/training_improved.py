#!/usr/bin/env python3
"""
Recommendation Model Training Pipeline
Memperbaiki: data mutation, error handling, hybrid recommendations
"""

import subprocess
import sys

# ============ INSTALL DEPENDENCIES ============
def install_dependencies():
    """Install required packages"""
    print("🔧 Installing required dependencies...")
    packages = [
        'boto3',
        'pandas',
        'numpy',
        'scikit-learn',
        'pyarrow',  # untuk read_parquet
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"✓ {package} installed")
    
    print("✓ All dependencies ready!\n")

# Install dependencies first
install_dependencies()

# ============ MAIN IMPORTS ============
import boto3
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
import json
from datetime import datetime
import logging

# ============ LOGGING SETUP ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============ INITIALIZE AWS CLIENTS ============
try:
    s3 = boto3.client('s3')
    dynamodb = boto3.resource('dynamodb')
    logger.info("AWS clients initialized successfully")
except Exception as e:
    logger.warning(f"AWS credentials not configured: {e}")
    s3 = None
    dynamodb = None

# ============ DATA LOADING ============
def load_data_from_s3():
    """Load processed data from S3"""
    if s3 is None:
        raise RuntimeError("AWS S3 client not initialized. Check credentials.")
    
    logger.info("Loading user_content_matrix...")
    user_content_df = pd.read_parquet('s3://streamify-bucket3-lks2026/processed-data/user_content_matrix/')
    
    logger.info("Loading content_stats...")
    content_stats_df = pd.read_parquet('s3://streamify-bucket3-lks2026/processed-data/content_stats/')
    
    logger.info("Loading user_features...")
    user_features_df = pd.read_parquet('s3://streamify-bucket3-lks2026/processed-data/user_features/')
    
    return user_content_df, content_stats_df, user_features_df

# ============ CONTENT-BASED RECOMMENDER ============
class ContentBasedRecommender:
    """Content-based filtering using TF-IDF"""
    
    def __init__(self):
        self.content_features = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.tfidf_vectorizer = None

    def fit(self, content_df):
        """Train content-based model"""
        # BUG FIX: Copy dataframe to avoid mutating input
        content_df = content_df.copy()
        
        # BUG FIX: Convert to string to avoid type errors
        content_df['content'] = (
            content_df['content_type'].astype(str) + ' ' +
            content_df['genre'].astype(str) + ' ' +
            content_df['title'].astype(str)
        )
        
        logger.info("Fitting TF-IDF vectorizer...")
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(content_df['content'])
        
        logger.info("Computing similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        self.content_features = content_df[['content_id', 'content_type', 'genre', 'title']]

    def recommend(self, content_id, n_recommendations=10):
        """Get similar content recommendations"""
        try:
            matching = self.content_features[self.content_features['content_id'] == content_id]
            if matching.empty:
                logger.warning(f"Content ID {content_id} not found")
                return []
            
            idx = matching.index[0]
            sim_scores = list(enumerate(self.similarity_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            similar_content = []
            for i, score in sim_scores[1:n_recommendations+1]:
                similar_content.append({
                    'content_id': self.content_features.iloc[i]['content_id'],
                    'similarity_score': float(score)
                })
            return similar_content
            
        except Exception as e:
            logger.error(f"Error in content-based recommendation: {e}")
            return []

# ============ COLLABORATIVE RECOMMENDER ============
class CollaborativeRecommender:
    """Collaborative filtering using SVD"""
    
    def __init__(self):
        self.user_content_matrix = None
        self.svd_model = None
        self.user_embeddings = None
        self.item_embeddings = None

    def fit(self, user_content_df):
        """Train collaborative filtering model"""
        logger.info("Creating user-content matrix...")
        self.user_content_matrix = user_content_df.pivot_table(
            index='user_id',
            columns='content_id',
            values='stream_count',
            fill_value=0
        )
        
        logger.info("Applying SVD decomposition...")
        self.svd_model = TruncatedSVD(n_components=50, random_state=42)
        self.user_embeddings = self.svd_model.fit_transform(self.user_content_matrix)
        self.item_embeddings = self.svd_model.components_.T
        logger.info(f"Matrix shape: {self.user_content_matrix.shape}")

    def recommend(self, user_id, n_recommendations=10):
        """Get recommendations for a user"""
        try:
            user_idx = self.user_content_matrix.index.get_loc(user_id)
            user_vector = self.user_embeddings[user_idx]
            scores = np.dot(user_vector, self.item_embeddings.T)
            top_items = np.argsort(scores)[::-1][:n_recommendations]
            
            recommendations = []
            for item_idx in top_items:
                content_id = self.user_content_matrix.columns[item_idx]
                score = scores[item_idx]
                recommendations.append({
                    'content_id': content_id,
                    'prediction_score': float(score)
                })
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in collaborative recommendation: {e}")
            return []

# ============ HYBRID RECOMMENDER ============
class HybridRecommender:
    """Hybrid approach combining content and collaborative filtering"""
    
    def __init__(self, content_weight=0.3, collab_weight=0.7):
        self.content_model = ContentBasedRecommender()
        self.collaborative_model = CollaborativeRecommender()
        self.content_weight = content_weight
        self.collab_weight = collab_weight

    def fit(self, user_content_df, content_df):
        """Train both models"""
        logger.info("Training content-based model...")
        self.content_model.fit(content_df)
        
        logger.info("Training collaborative model...")
        self.collaborative_model.fit(user_content_df)
        logger.info("✓ Hybrid model training complete")

    def recommend(self, user_id, n_recommendations=10, seed_content_id=None):
        """Get hybrid recommendations combining both models"""
        collab_recs = self.collaborative_model.recommend(user_id, n_recommendations)
        if not collab_recs:
            logger.warning(f"No collaborative recommendations for user {user_id}")
            return []
        
        final_recs = []
        seen_content = set()
        
        # BUG FIX: Blend content-based and collaborative scores properly
        for rec in collab_recs[:n_recommendations]:
            content_id = rec['content_id']
            
            # Get content-based score if seed content provided
            content_score = 0.0
            if seed_content_id:
                similar = self.content_model.recommend(seed_content_id, n_recommendations=100)
                matching = [s for s in similar if s['content_id'] == content_id]
                if matching:
                    content_score = matching[0]['similarity_score']
            
            # Blend scores properly
            blended_score = (
                self.collab_weight * rec['prediction_score'] + 
                self.content_weight * content_score
            )
            
            if content_id not in seen_content:
                final_recs.append({
                    'content_id': content_id,
                    'score': blended_score,
                    'type': 'hybrid',
                    'collab_score': rec['prediction_score'],
                    'content_score': content_score
                })
                seen_content.add(content_id)
        
        # Sort by blended score
        final_recs = sorted(final_recs, key=lambda x: x['score'], reverse=True)[:n_recommendations]
        return final_recs

# ============ MAIN TRAINING FUNCTION ============
def train_recommendation_model():
    """Main training pipeline"""
    try:
        logger.info("=" * 50)
        logger.info("Starting Recommendation Model Training")
        logger.info("=" * 50)
        
        # Load data
        logger.info("Loading data from S3...")
        user_content_df, content_stats_df, user_features_df = load_data_from_s3()
        logger.info(f"✓ Loaded user_content: {user_content_df.shape}")
        logger.info(f"✓ Loaded content_stats: {content_stats_df.shape}")
        logger.info(f"✓ Loaded user_features: {user_features_df.shape}")

        # Load content catalog
        logger.info("Loading content catalog...")
        content_df = pd.read_csv('s3://streamify-bucket3-lks2026/raw-data/content-catalog/content_catalog.csv')
        logger.info(f"✓ Loaded catalog: {content_df.shape}")

        # Train model
        logger.info("Training hybrid model...")
        hybrid_model = HybridRecommender()
        hybrid_model.fit(user_content_df, content_df)

        # Save model
        logger.info("Saving model to S3...")
        with open('/tmp/hybrid_model.pkl', 'wb') as f:
            pickle.dump(hybrid_model, f)
        
        if s3 is not None:
            s3.upload_file('/tmp/hybrid_model.pkl', 'streamify-bucket3-lks2026', 'models/hybrid_model.pkl')
            logger.info("✓ Model saved to S3")

        # Generate embeddings
        logger.info("Generating content embeddings...")
        if dynamodb is None:
            raise RuntimeError("DynamoDB not initialized")
        
        content_embeddings_table = dynamodb.Table('ContentEmbeddings')

        # Determine streams column
        logger.info(f"Available columns: {content_stats_df.columns.tolist()}")
        
        streams_col = None
        possible_names = ['total_streams', 'stream_count', 'total_streams_count', 'plays', 'total_plays', 'streams']
        for col in possible_names:
            if col in content_stats_df.columns:
                streams_col = col
                break
        
        if streams_col is None:
            logger.warning("No stream count column found. Using default 0.")
            content_stats_df['total_streams'] = 0
            streams_col = 'total_streams'
        else:
            logger.info(f"Using column '{streams_col}' for popularity")
        
        # Process each content item
        total_items = len(content_df)
        for idx, row in content_df.iterrows():
            content_id = row['content_id']
            
            matching = content_stats_df[content_stats_df['content_id'] == content_id]
            popularity = float(matching[streams_col].iloc[0]) if not matching.empty else 0.0
            
            embedding = {
                'content_type': str(row.get('content_type', '')),
                'genre': str(row.get('genre', '')),
                'is_exclusive': bool(row.get('is_exclusive', False)),
                'popularity': popularity
            }
            
            content_embeddings_table.put_item(
                Item={
                    'content_id': content_id,
                    'embedding': json.dumps(embedding),
                    'last_updated': datetime.now().isoformat()
                }
            )
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{total_items} items")
        
        logger.info("=" * 50)
        logger.info("✓ Model training completed successfully!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

# ============ EXECUTE ============
if __name__ == "__main__":
    train_recommendation_model()
