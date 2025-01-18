import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


movie_ids_titles = pd.read_csv("movies.csv")
#print(movie_ids_titles.shape)
movie_ids_ratings = pd.read_csv("ratings.csv")
#print(movie_ids_ratings.shape)

movie_ids_titles.drop(['genres'], inplace=True, axis=1)

#print(movie_ids_titles.head())
movie_ids_ratings.drop(["timestamp"], inplace=True, axis=1)

#print(movie_ids_ratings.head())
merged_movie_df = pd.merge(movie_ids_ratings, movie_ids_titles, on='movieId')

#print(merged_movie_df.head())

#print(merged_movie_df.groupby('title').describe())
#print(merged_movie_df.groupby('title')['rating'].mean().head())
#print(merged_movie_df.groupby('title')['rating'].mean().sort_values(ascending=False).head())
#print(merged_movie_df.groupby('title')['rating'].count().sort_values(ascending=False).head())
movie_rating_mean_count = pd.DataFrame(columns=['rating_mean', 'rating_count'])
movie_rating_mean_count["rating_mean"] = merged_movie_df.groupby('title')['rating'].mean()
movie_rating_mean_count["rating_count"] = merged_movie_df.groupby('title')['rating'].count()
print(movie_rating_mean_count.head())

plt.figure()
sns.set_style("dark")
movie_rating_mean_count['rating_mean'].hist(bins=30, color='red')
movie_rating_mean_count['rating_count'].hist(bins=30, color='black')
plt.figure(figsize=(10, 8))
sns.set_style("darkgrid")

sns.regplot(x="rating_mean", y="rating_count", data=movie_rating_mean_count, color="brown")
plt.show()
print(movie_rating_mean_count.sort_values("rating_count", ascending=False).head())
user_movie_rating_matrix = merged_movie_df.pivot_table(index="userId", columns="title", values="rating")
print(user_movie_rating_matrix)

print(user_movie_rating_matrix.shape)
print(user_movie_rating_matrix)
pulp_fiction_ratings = user_movie_rating_matrix["Pulp Fiction (1994)"]
pulp_fiction_correlations = pd.DataFrame(user_movie_rating_matrix.corrwith(pulp_fiction_ratings),columns=["pf_corr"])
print(pulp_fiction_correlations.sort_values("pf_corr", ascending=False).head(5))



pulp_fiction_correlations = pulp_fiction_correlations.join(movie_rating_mean_count["rating_count"])

print(pulp_fiction_correlations.head())
pulp_fiction_correlations.dropna(inplace=True)
print(pulp_fiction_correlations.sort_values("pf_corr", ascending=False).head())
pulp_fiction_correlations_50 = pulp_fiction_correlations[pulp_fiction_correlations['rating_count'] > 50]

print(pulp_fiction_correlations_50.sort_values("pf_corr", ascending=False).head())
all_movie_correlations = user_movie_rating_matrix.corr(method="pearson",min_periods=50)
print(all_movie_correlations.head())
movie_data = [['Forrest Gump (1994)', 4.0], ['Fight Club (1999)', 3.5], ['Interstellar (2014)', 4.0]]

test_movies = pd.DataFrame(movie_data, columns=['Movie_Name', 'Movie_Rating'])

print(test_movies.head())
print(test_movies['Movie_Name'][0])
print(test_movies['Movie_Rating'][0])
print(all_movie_correlations['Forrest Gump (1994)'].dropna())


# Initialize recommended_movies as an empty list instead of a Series
recommended_movies = []

# Loop through movies to generate recommendations
for i in range(0, 2):
    movie = all_movie_correlations[test_movies['Movie_Name'][i]].dropna()
    movie = movie.map(lambda movie_corr: movie_corr *
                      test_movies["Movie_Rating"][i])

    # Append only non-empty results to the list
    if not movie.empty:
        recommended_movies.append(movie)

# Check if we have any recommendations
if recommended_movies:
    # Flatten the list of Series into a single Series
    flattened_recommendations = pd.concat(recommended_movies)

    # Sort the flattened recommendations
    sorted_recommendations = flattened_recommendations.sort_values(
        ascending=False)
    print(sorted_recommendations.head(10))
else:
    print("No recommendations available.")

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
movie_ids_titles = pd.read_csv("movies.csv")
movie_ids_ratings = pd.read_csv("ratings.csv")

# Drop unnecessary columns
movie_ids_titles.drop(['genres'], inplace=True, axis=1)
movie_ids_ratings.drop(["timestamp"], inplace=True, axis=1)

# Merge movie data
merged_movie_df = pd.merge(movie_ids_ratings, movie_ids_titles, on='movieId')

# Calculate mean and count of ratings for each movie
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
        print("Recommended movies based on your favorite:")
        print(sorted_recommendations.head(10))
    else:
        print("No recommendations available for the selected movie.")
else:
    print("Movie not found in the dataset. Please try a different title.")



# Load data
movie_ids_titles = pd.read_csv("movies.csv")
movie_ids_ratings = pd.read_csv("ratings.csv")

# Drop unnecessary columns
movie_ids_titles.drop(['genres'], inplace=True, axis=1)
movie_ids_ratings.drop(["timestamp"], inplace=True, axis=1)

# Merge movie data
merged_movie_df = pd.merge(movie_ids_ratings, movie_ids_titles, on='movieId')

# Calculate mean and count of ratings for each movie
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


'''
