import streamlit as st
import pandas as pd
import joblib

# Load preprocessed data
df = pd.read_csv('preprocessed_movies.csv')

# Load scaler and model
scaler = joblib.load('scaler.pkl')
knn = joblib.load('knn_model.pkl')

st.title('Movie Recommendation System')

# User input
user_id = st.number_input('Enter User ID', min_value=1, max_value=df['userId'].max(), value=1)
num_suggestions = st.slider('Number of Suggestions', min_value=1, max_value=20, value=5)

# Prepare input features
user_data = df[df['userId'] == user_id]

if user_data.empty:
    st.write(f"No data found for User ID {user_id}.")
else:
    user_data = user_data.iloc[0]
    user_features = pd.DataFrame([user_data[['userId', 'age', 'occupation', 'rating', 'gender'] + list(df.columns[df.columns.str.startswith('Action')])].values],
                                 columns=['userId', 'age', 'occupation', 'rating', 'gender'] + list(df.columns[df.columns.str.startswith('Action')]))

    # Standardize and predict
    user_features_scaled = scaler.transform(user_features)
    distances, indices = knn.kneighbors(user_features_scaled, n_neighbors=num_suggestions)

    # Retrieve movie IDs from indices and ensure uniqueness
    suggested_movie_ids = list(set(df.iloc[indices[0]]['movieId'].tolist()))

    # Retrieve and display suggestions
    suggested_movies = df[df['movieId'].isin(suggested_movie_ids)].drop_duplicates(subset=['movieId'])

    if suggested_movies.empty:
        st.write("No recommendations available.")
    else:
        st.write(f"Recommended movies for User ID {user_id}:")
        st.dataframe(suggested_movies[['title', 'genres']])
