import streamlit as st
import pandas as pd
import numpy as np

np.seterr(divide='ignore', invalid='ignore')

st.write("""

# BDAA Group 11 SVD Recommender System Working Demo with GUI
Fan Yue, 17220701\n
Patrick Loh Song Ta, S2021031\n
Norzarifah Kamarauzaman, 17041741\n
Shaniza Binti Meor Danial, WQD180118\n
Lim Wen Yan, S2019787\n
Sunny Chan Zi Yang, S2022037\n\n\n




            """)



# read dataset (MovieLens 1M movie ratings dataset: downloaded from https://grouplens.org/datasets/movielens/1m/)
data = pd.io.parsers.read_csv('ratings.dat', 
    names=['user_id', 'movie_id', 'rating', 'time'],
    engine='python', delimiter='::')
movie_data = pd.io.parsers.read_csv('movies.dat',
    names=['movie_id', 'title', 'genre'],
    engine='python', delimiter='::')

df = pd.DataFrame(movie_data[['movie_id', 'title']])

st.sidebar.dataframe(df)


# create the rating matrix (rows as movies, columns as users)

# create an empty array
ratings_mat = np.ndarray(shape=(np.max(data['movie_id'].values), np.max(data['user_id'].values)), dtype=np.uint8)

# fill in the array
ratings_mat[data["movie_id"].values-1, data["user_id"].values-1] = data["rating"].values

# normalize the matrix (subtract mean off)
normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T

# computing the Singular Value Decomposition (SVD)
A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
U, S, V = np.linalg.svd(A)




# define function to calculate the cosine similarity (sorting by most similar and returning the top N)
def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1 # Movie id starts from 1 in the dataset
    movie_row = data[index, :]

    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))

    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

# define function to print top N similar movies
def print_similar_movies(movie_data, movie_id, top_indexes):
    
    print('Recommendations for {0}: \n'.format(
    movie_data[movie_data.movie_id == movie_id].title.values[0]))

    st.write('Recommendations for {0}: \n'.format(
    movie_data[movie_data.movie_id == movie_id].title.values[0]))

    for id in top_indexes + 1:
        print(movie_data[movie_data.movie_id == id].title.values[0])
        st.write(movie_data[movie_data.movie_id == id].title.values[0])



add_slider = st.sidebar.slider(
    'Select a movie ID:',
    1, 3952, (2)
)



st.write('**The chosen Movie ID is **', add_slider )

# k-principal components to represent movies, movie_id to find recommendations, top_n print n results        
k = 50
movie_id = add_slider #10 # (getting an id from movies.dat)
top_n = 10
sliced = V.T[:, :k] # representative data
indexes = top_cosine_similarity(sliced, movie_id, top_n)

print_similar_movies(movie_data, movie_id, indexes)










