import streamlit as st
import pickle
import pandas as pd
import requests
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Initialize session state for number of posters to show
if 'num_posters' not in st.session_state:
    st.session_state.num_posters = 5

# Load data first
try:
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('similarity.pkl', 'rb'))
except FileNotFoundError:
    st.error("Pickle files not found. Please ensure 'movie_dict.pkl' and 'similarity.pkl' are in the same directory.")
    st.stop()


with open('movie_dict.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
# print(loaded_dict.keys())



# Function to fetch movie poster and vote_average
def fetch_poster_and_rating(id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        vote_average = data.get('vote_average', None)
        if poster_path:
            poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path}"
        else:
            poster_url = "https://via.placeholder.com/500x750?text=No+Poster"
        return poster_url, vote_average
    except:
        return "https://via.placeholder.com/500x750?text=Error", None

# Tag-based recommendation
def recommend_tag_based(movie, n=5, similarity_matrix=similarity, df=movies):
    movie_index = df[df['title'].str.lower() == movie.lower()].index
    if len(movie_index) == 0:
        return "Movie not found", []
    movie_index = movie_index[0]
    distances = sorted(list(enumerate(similarity_matrix[movie_index])), reverse=True, key=lambda x: x[1])[1:n+1]
    recommended_movie_names = [df.iloc[i[0]].title for i in distances]
    recommended_movie_posters = [fetch_poster_and_rating(df.iloc[i[0]]['id'])[0] for i in distances]
    return recommended_movie_names, recommended_movie_posters

# Genre-based recommendation
def recommend_genre_based(movie, n=5, df=movies):
    movie_index = df[df['title'].str.lower() == movie.lower()].index
    if len(movie_index) == 0:
        return "Movie not found", []
    movie_index = movie_index[0]
    genre_tags = ' '.join(df.iloc[movie_index]['genres']).lower()
    cv_genre = CountVectorizer(stop_words='english')
    genre_vectors = cv_genre.fit_transform([genre_tags] + df['genres'].apply(lambda x: ' '.join(x)).tolist()).toarray()
    genre_similarity = cosine_similarity(genre_vectors)[0, 1:]
    movies_list = sorted(list(enumerate(genre_similarity)), reverse=True, key=lambda x: x[1])[:n]
    recommended_movie_names = [df.iloc[i[0]].title for i in movies_list]
    recommended_movie_posters = [fetch_poster_and_rating(df.iloc[i[0]]['id'])[0] for i in movies_list]
    return recommended_movie_names, recommended_movie_posters

# Recommend movies by genre
def recommend_by_genre(genre, n=5, df=movies):
    genre = genre.lower()
    # Find all movies that have this genre
    genre_mask = df['genres'].apply(lambda genres: genre in [g.lower() for g in genres])
    genre_movies = df[genre_mask]
    if genre_movies.empty:
        return [], []
    # Sort by rating if available
    if 'vote_average' in genre_movies.columns:
        genre_movies = genre_movies.sort_values(by='vote_average', ascending=False)
    top_movies = genre_movies.head(n)
    names = top_movies['title'].tolist()
    posters = [fetch_poster_and_rating(mid)[0] for mid in top_movies['id']]
    return names, posters

# Evaluation function
def evaluate_recommendations(movie, recommend_func, n=5, genre_overlap_threshold=0.75, df=movies):
    if movie.lower() not in df['title'].str.lower().values:
        return None
    
    input_genres = set(df[df['title'].str.lower() == movie.lower()]['genres'].iloc[0])
    if not input_genres:
        return None
    
    y_true = []
    for idx, row in df.iterrows():
        if row['title'].lower() == movie.lower():
            continue
        other_genres = set(row['genres'])
        if not other_genres:
            continue
        overlap = len(input_genres & other_genres) / len(input_genres)
        if overlap >= genre_overlap_threshold:
            y_true.append(row['title'])
    
    if not y_true:
        return None
    
    y_pred, _ = recommend_func(movie, n)
    if not isinstance(y_pred, list):
        return None
    
    all_movies = list(set(y_true + y_pred))
    y_true_binary = [1 if m in y_true else 0 for m in all_movies]
    y_pred_binary = [1 if m in y_pred else 0 for m in all_movies]
    
    precision = precision_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_recommended': len(y_pred),
        'n_ground_truth': len(y_true)
    }



# Professional Debug Information Display
st.sidebar.markdown("### Debug Information")
st.sidebar.write(f"Total number of movies: {len(movies)}")

# Removed the columns listing section entirely

st.sidebar.write("Sample movie titles:")
for title in movies['title'].head(5):
    st.sidebar.write(f"• {title}")

st.sidebar.write(f"Is 'Avatar' present in the dataset? {'Yes' if 'Avatar' in movies['title'].values else 'No'}")


# Ensure required columns
required_columns = ['id', 'title', 'genres']
if not all(col in movies.columns for col in required_columns):
    st.error(f"Required columns missing in movie_dict.pkl. Expected: {required_columns}, Found: {movies.columns.tolist()}")
    st.stop()

# Function to show default recommendations
def show_default_recommendations(n=5, df=movies):
    # Fetch vote_average if not present and sort by it
    if 'vote_average' not in df.columns:
        st.warning("Fetching vote_average from TMDB API for all movies...")
        df['vote_average'] = [fetch_poster_and_rating(mid)[1] for mid in df['id']]
    top_movies = df.sort_values(by='vote_average', ascending=False).head(n) if 'vote_average' in df.columns else df.sample(n)
    names = top_movies['title'].tolist()
    posters = [fetch_poster_and_rating(mid)[0] for mid in top_movies['id']]
    return "Popular Movies", names, posters

# Streamlit UI
st.header('Movie Recommender System')

search_query = st.text_input("Enter movie name or genre", "")

# Find all matching movie names and genres after search
matches = []
if search_query:
    # Movie title matches (starts with)
    movie_matches = movies[movies['title'].str.lower().str.startswith(search_query.lower())]['title'].tolist()
    matches.extend(movie_matches)
    # Genre matches (starts with)
    all_genres = set([genre for sublist in movies['genres'] for genre in sublist])
    genre_matches = [g for g in all_genres if g.lower().startswith(search_query.lower())]
    matches.extend([f"Genre: {g}" for g in genre_matches])

if st.button('Show Recommendations'):
    n = 10  # Show 10 posters
    recommended_movie_names = []
    recommended_movie_descriptions = []
    if not search_query:
        header, recommended_movie_names, recommended_movie_posters = show_default_recommendations(n=n)
        recommended_movie_descriptions = [
            movies[movies['title'] == name]['overview'].iloc[0]
            if 'overview' in movies.columns and not movies[movies['title'] == name].empty
            else 'No description available.'
            for name in recommended_movie_names
        ]
    else:
        if not matches:
            st.warning("No movies or genres found matching your search.")
            recommended_movie_posters = []
        else:
            match_to_use = matches[0]
            if match_to_use.startswith('Genre:'):
                # Handle genre-based recommendation (improved)
                genre_name = match_to_use.replace('Genre: ', '')
                recommended_movie_names, recommended_movie_posters = recommend_by_genre(genre_name, n=n)
            else:
                # Handle movie-based recommendation
                recommended_movie_names, recommended_movie_posters = recommend_tag_based(match_to_use, n=n)
            recommended_movie_descriptions = [
                movies[movies['title'] == name]['overview'].iloc[0]
                if 'overview' in movies.columns and not movies[movies['title'] == name].empty
                else 'No description available.'
                for name in recommended_movie_names
            ]
    if recommended_movie_posters:
        num_to_show = st.session_state.num_posters
        cols = st.columns(5)
        for i in range(min(num_to_show, len(recommended_movie_posters))):
            with cols[i % 5]:
                # Get movie details for ID, year, rating, and genres
                movie_row = movies[movies['title'] == recommended_movie_names[i]]
                movie_id = movie_row['id'].iloc[0] if not movie_row.empty else 'N/A'
                # Shorten title to max 3 words
                short_title = ' '.join(recommended_movie_names[i].split()[:3])
                # Get full title
                full_title = recommended_movie_names[i]
                # Get release year if available
                release_year = 'N/A'
                if not movie_row.empty and 'release_date' in movie_row.columns:
                    date_val = movie_row['release_date'].iloc[0]
                    if isinstance(date_val, str) and len(date_val) >= 4:
                        release_year = date_val[:4]
                # Get rating if available
                rating = movie_row['vote_average'].iloc[0] if not movie_row.empty and 'vote_average' in movie_row.columns else 'N/A'
                # Get genres if available
                genres = ', '.join(movie_row['genres'].iloc[0]) if not movie_row.empty and 'genres' in movie_row.columns and isinstance(movie_row['genres'].iloc[0], list) else 'N/A'
                # Show short movie title
                st.markdown(f"<div style='font-size: 1.2em; font-weight: bold;'>{short_title}</div>", unsafe_allow_html=True)
                # Show poster
                st.image(recommended_movie_posters[i])
                # Show full movie title
                st.caption(f"Title: {full_title}")
                # Show movie ID
                st.caption(f"ID: {movie_id}")
                # Show release year if available
                if release_year != 'N/A':
                    st.caption(f"Year: {release_year}")
                # Show rating if available
                if rating != 'N/A':
                    st.caption(f"Rating: {rating}")
                # Show genres if available
                if genres != 'N/A':
                    st.caption(f"Genres: {genres}")
        if num_to_show < len(recommended_movie_posters):
            if st.button("Show More"):
                st.session_state.num_posters += 5

# Visualization functions
def plot_genre_distribution():
    """Plot the top 10 movie genres distribution"""
    try:
        # Extract all genres from the nested lists
        all_genres = [genre for sublist in movies['genres'] for genre in sublist]
        genre_counts = pd.Series(all_genres).value_counts()
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x=genre_counts.values[:10], y=genre_counts.index[:10], palette='viridis')
        plt.title('Top 10 Movie Genres Distribution')
        plt.xlabel('Number of Movies')
        plt.ylabel('Genre')
        
        # Save and display
        plt.tight_layout()
        plt.savefig('genre_distribution.png')
        st.pyplot(plt.gcf())
        plt.close()
    except Exception as e:
        st.error(f"Error generating genre distribution: {str(e)}")

def plot_rating_distribution():
    """Plot the distribution of movie ratings if available"""
    try:
        rating_cols = ['vote_average', 'rating', 'score', 'vote', 'imdb_rating', 'tmdb_rating']
   
        rating_col = next((col for col in rating_cols if col in movies.columns), None)
        
        if rating_col is None:
            st.warning("No rating data available to display. Tried columns: " + ", ".join(rating_cols))
            return
        rating_data = movies[rating_col].dropna()
        if len(rating_data) == 0:
            st.warning(f"Rating column '{rating_col}' exists but contains no data")
            return
        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.histplot(rating_data, bins=20, kde=True, color='blue')
        plt.title(f'Distribution of Movie Ratings ({rating_col})')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        st.pyplot(plt.gcf())
        plt.close()
    except Exception as e:
        st.error(f"Error generating rating distribution: {str(e)}")

# Visualization section
if st.sidebar.button('Generate Dataset Visualizations'):
    st.subheader("Dataset Visualizations")
    
    # Plot genre distribution
    with st.spinner('Generating genre distribution...'):
        plot_genre_distribution()
    
    # Plot rating distribution
    with st.spinner('Generating rating distribution...'):
        plot_rating_distribution()
    
    st.success("Visualizations generated successfully!")