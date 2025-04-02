import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
movie_ids_titles = pd.read_csv("movies.csv")
movie_ids_ratings = pd.read_csv("ratings.csv")


movie_ids_titles.drop(['genres'], inplace=True, axis=1)
movie_ids_ratings.drop(["timestamp"], inplace=True, axis=1)


merged_movie_df = pd.merge(movie_ids_ratings, movie_ids_titles, on='movieId')


movie_rating_mean_count = pd.DataFrame(columns=['rating_mean', 'rating_count'])
movie_rating_mean_count["rating_mean"] = merged_movie_df.groupby('title')[
    'rating'].mean()
movie_rating_mean_count["rating_count"] = merged_movie_df.groupby('title')[
    'rating'].count()

# Create user-movie rating matrix
user_movie_rating_matrix = merged_movie_df.pivot_table(
    index="userId", columns="title", values="rating")

# Correlation matrix with minimum 50 reviews
all_movie_correlations = user_movie_rating_matrix.corr(
    method="pearson", min_periods=50)

# User input for favorite movie title
user_movie = input(
    "Enter your favorite movie title (exactly as in the dataset): ")
user_rating = float(input("Rate this movie (e.g., 4.0): "))

# Recommendations
recommended_movies = []

# Check if the movie is in the dataset
if user_movie in all_movie_correlations.columns:
    # Get the correlations of the specified movie
    movie_correlations = all_movie_correlations[user_movie].dropna()
    movie_correlations = movie_correlations.map(
        lambda movie_corr: movie_corr * user_rating)
    recommended_movies.append(movie_correlations)

    # Flatten and sort the recommendations
    if recommended_movies:
        flattened_recommendations = pd.concat(recommended_movies)
        sorted_recommendations = flattened_recommendations.sort_values(
            ascending=False)

        # Calculate probability of liking each recommended movie
        probabilities = (sorted_recommendations / user_rating) * 100

        # Display results
        print("Recommended movies based on your favorite:")
        for title, score in sorted_recommendations.head(10).items():
            probability = probabilities[title] if title in probabilities else 0
            print(f"{title}: {score:.2f} (Probability of liking: {probability:.2f}%)")
    else:
        print("No recommendations available for the selected movie.")
else:
    print("Movie not found in the dataset. Please try a different title.")
