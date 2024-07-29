import streamlit as st
import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD
from surprise import accuracy

# Load the model
with open('svd_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load preprocessed movie data
movies = pd.read_csv('preprocessed_movies.csv')

# Streamlit app
st.title('Movie Recommendation System')

# User input
user_id = st.number_input('Enter User ID', min_value=1, max_value=int(movies['userId'].max()), value=1)
num_recommendations = st.slider('Number of Recommendations', min_value=1, max_value=10, value=5)

# Prepare the data for Surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(movies[['userId', 'movieId', 'rating']], reader)

# Get movie IDs
movie_ids = movies['movieId'].unique()

# Predict ratings for all movies for the given user
predictions = []
for movie_id in movie_ids:
    pred = model.predict(user_id, movie_id)
    predictions.append((movie_id, pred.est))

# Sort predictions by estimated rating
predictions.sort(key=lambda x: x[1], reverse=True)

# Get top N recommendations
top_predictions = predictions[:num_recommendations]

# Display recommendations
st.subheader('Top Recommendations:')
for movie_id, rating in top_predictions:
    movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
    st.write(f"{movie_title} (Predicted Rating: {rating:.2f})")
