import pandas as pd
import streamlit as st
import joblib

# Load the preprocessed dataset
df = pd.read_csv('preprocessed_movies.csv')

# Define feature columns
feature_cols = ['age', 'occupation', 'gender'] + [col for col in df.columns if col.startswith('Action') or col.startswith('Adventure') or col.startswith('Animation') or col.startswith('Children') or col.startswith('Comedy') or col.startswith('Crime') or col.startswith('Documentary') or col.startswith('Drama') or col.startswith('Fantasy') or col.startswith('Film-Noir') or col.startswith('Horror') or col.startswith('IMAX') or col.startswith('Musical') or col.startswith('Mystery') or col.startswith('Romance') or col.startswith('Sci-Fi') or col.startswith('Thriller') or col.startswith('War') or col.startswith('Western')]

# Load the saved model
model = joblib.load('gradient_boosting_model.pkl')

# Streamlit interface
st.title('Movie Recommendation System')

# User inputs
user_id = st.number_input('Enter User ID', min_value=1, max_value=df['userId'].max(), value=1)
num_recommendations = st.slider('Number of Recommendations', min_value=1, max_value=20, value=5)

if st.button('Get Recommendations'):
    user_data = df[df['userId'] == user_id]
    
    if not user_data.empty:
        # Extract the user's features
        user_features = user_data[feature_cols].mean().values.reshape(1, -1)
        
        # Prepare features for all movies
        all_movies = df[['movieId', 'title']].drop_duplicates()
        
        # Create a DataFrame to hold movie features
        all_movies_features = pd.DataFrame(columns=feature_cols)
        
        # Populate the DataFrame with movie features
        movie_features_list = []
        for movie_id in all_movies['movieId']:
            movie_features = df[df['movieId'] == movie_id][feature_cols].mean()
            movie_features_list.append(pd.Series(movie_features, name=movie_id))
        
        all_movies_features = pd.concat(movie_features_list, axis=1).T
        
        # Reset the index to ensure movieId matches properly
        all_movies_features.reset_index(drop=True, inplace=True)
        
        # Add movieId and title columns
        all_movies_features['movieId'] = all_movies['movieId'].values
        all_movies_features['title'] = all_movies['title'].values
        
        # Ensure the correct order of features for prediction
        all_movies_features = all_movies_features[feature_cols]
        
        # Add user-specific features
        for feature in ['age', 'occupation', 'gender']:
            all_movies_features[feature] = user_features[0][feature_cols.index(feature)]
        
        # Predict ratings for all movies
        predicted_ratings = model.predict(all_movies_features)
        
        # Add predictions to the DataFrame
        all_movies_features['predicted_rating'] = predicted_ratings
        
        # Merge the title and movieId back into the DataFrame
        recommendations = pd.merge(all_movies, all_movies_features[['movieId', 'predicted_rating']], on='movieId')
        recommendations = recommendations.sort_values(by='predicted_rating', ascending=False).head(num_recommendations)
        
        st.write('Recommended Movies:')
        st.write(recommendations[['title', 'predicted_rating']])
    else:
        st.write('No data found for this User ID')
