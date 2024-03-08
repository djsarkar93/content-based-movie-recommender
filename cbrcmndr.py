########################################################################################################################
# Imports
########################################################################################################################
import ast
import pickle

import numpy as np
import pandas as pd

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



########################################################################################################################
# Reading the datasets
########################################################################################################################
movies_df = pd.read_csv('datasets/tmdb_5000_movies.csv')
#print(movies_df.shape)
#print(movies_df.head(3))

credits_df = pd.read_csv('datasets/tmdb_5000_credits.csv')
#print(credits_df.shape)
#print(credits_df.head(3))



########################################################################################################################
# Joining the 2 dataframes
########################################################################################################################
movies_credits_df = pd.merge(movies_df, credits_df, left_on='id', right_on='movie_id')
movies_credits_df = movies_credits_df.rename(columns={'title_x': 'title'})
movies_credits_df = movies_credits_df.drop('title_y', axis=1)
#print(movies_credits_df.head(3))



########################################################################################################################
# Selecting specific columns from the dataset. These columus will help us build tags for the model.
########################################################################################################################
movies_refined_df = movies_credits_df[['id','title','overview','genres','keywords','cast','crew']]
#print(movies_refined_df.head(3))



########################################################################################################################
# Preprocessing the data
########################################################################################################################
#NULL handling
#print(movies_refined_df.isnull().sum())
#Only overview has null and that too just in 3 records.
movies_refined_df.dropna(inplace=True)

#De-duplicating
#print(movies_refined_df.duplicated().sum())

#Formatting the genres column
#print(movies_refined_df[['genres']].iloc[0].values)
movies_refined_df['genres'] = movies_refined_df['genres'].apply(lambda obj: [   i['name'].replace(' ', '') 
                                                                                for i in ast.literal_eval(obj)  ])
#print(movies_refined_df.head(3))

#Formatting the keywords column
#print(movies_refined_df[['keywords']].iloc[0].values)
movies_refined_df['keywords'] = movies_refined_df['keywords'].apply(lambda obj: [   i['name'].replace(' ', '') 
                                                                                    for i in ast.literal_eval(obj)  ])
#print(movies_refined_df.head(3))

#Formatting the cast column
#print(movies_refined_df[['cast']].iloc[0].values)
movies_refined_df['cast'] = movies_refined_df['cast'].apply(lambda obj: [   i['name'].replace(' ', '') 
                                                                            for i in ast.literal_eval(obj)[:3]  ])
#print(movies_refined_df.head(3))

#Formatting the crew column
#print(movies_refined_df[['crew']].iloc[0].values)
movies_refined_df['crew'] = movies_refined_df['crew'].apply(lambda obj: [   i['name'].replace(' ', '') 
                                                                            for i in ast.literal_eval(obj)
                                                                            if i['job'] == 'Director'       ])
#print(movies_refined_df.head(3))

#Formatting the overview column
#print(movies_refined_df[['overview']].iloc[0].values)
movies_refined_df['overview'] = movies_refined_df['overview'].apply(lambda obj: obj.split())
#print(movies_refined_df.head(3))

#Creating the tags
movies_refined_df['tags'] = movies_refined_df['overview'] + movies_refined_df['genres'] + movies_refined_df['keywords'] + movies_refined_df['cast'] + movies_refined_df['crew']

movies_tags_df = movies_refined_df[['id','title','tags']]
movies_tags_df['tags'] = movies_tags_df['tags'].apply(lambda obj: ' '.join(obj).lower())
#print(movies_tags_df.head(3))

#Stemming the tags
#print(movies_tags_df[['tags']].iloc[0].values)

ps = PorterStemmer()
movies_tags_df['tags'] = movies_tags_df['tags'].apply(lambda obj: ' '.join([ps.stem(i) for i in obj.split()]))

#print(movies_tags_df[['tags']].iloc[0].values)
#print(movies_tags_df.head(3))



########################################################################################################################
# Vectorizing the tags column
########################################################################################################################
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies_tags_df['tags']).toarray()

#print(cv.get_feature_names_out().tolist())
#print(vectors)



########################################################################################################################
# Finding the cosine similarity between all the vectors
########################################################################################################################
similarity_matrix = cosine_similarity(vectors)

#print(similarity_matrix[1])
#print(similarity_matrix[2])
#print(similarity_matrix)



########################################################################################################################
# Creating a recommendation function
########################################################################################################################
def recommend(movie_title):
    movie_idx = movies_tags_df[ movies_tags_df['title'] == movie_title ].index[0]
    movie_similarity_scores = similarity_matrix[ movie_idx ]
    recommended_movie_ids = []
    for i in sorted(enumerate(movie_similarity_scores), key = lambda x: x[-1], reverse = True)[1:6]:
        recommended_movie_ids.append(i[0])
    return recommended_movie_ids

#print(recommend(movie_title='Batman Begins'))