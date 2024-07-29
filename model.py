import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load the datasets
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
users = pd.read_csv('ml-latest-small/users.csv')  # Make sure this path is correct

# Combine datasets
df = pd.merge(ratings, movies, on='movieId')
df = pd.merge(df, users, on='userId')

# Data cleaning
df.drop_duplicates(inplace=True)
df.columns = df.columns.str.strip()
df = df.drop(columns="timestamp", errors='ignore')
df['rating'].fillna(df['rating'].mean(), inplace=True)
df['rating'] = df['rating'].astype(int)

# Extract year from title and clean title
df['year'] = df['title'].str.extract(r'\((\d{4})\)').fillna(0).astype(int)
df['title'] = df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()

# Ensure 'genres' column is in string format
df['genres'] = df['genres'].astype(str)

# Check and drop existing genre columns if present
existing_genre_columns = [col for col in df.columns if col in ['(no genres listed)', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']]
df = df.drop(columns=existing_genre_columns, errors='ignore')

# One-hot encoding for genres
genres_dummies = df['genres'].str.get_dummies(sep='|')
df = df.join(genres_dummies)

# Lowercase the titles
df['title'] = df['title'].str.lower()

# Clip ratings to be within 1-5
df['rating'] = df['rating'].clip(1, 5)

# Drop columns with no genres listed
df = df.drop(columns="(no genres listed)", errors='ignore')

# Feature-based model data
genre_dummies = df.filter(like='genre_')
user_features = df[['userId', 'age', 'gender', 'occupation']]  # Include additional user features
user_features = pd.get_dummies(user_features, columns=['gender', 'occupation'], drop_first=True)  # One-hot encode categorical features
movie_features = df[['movieId']].join(genre_dummies)
features = pd.merge(user_features, movie_features, left_on='userId', right_index=True)
features = features.drop(columns=['movieId'])

# Define the target variable
target = df['rating']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Gradient Boosting Regressor model
feature_model = GradientBoostingRegressor()
feature_model.fit(X_train, y_train)

# Save the feature-based model
with open('feature_based_model.pkl', 'wb') as f:
    pickle.dump(feature_model, f)

# Evaluate the feature-based model
y_pred = feature_model.predict(X_test)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print('Feature-Based Model RMSE:', rmse)
