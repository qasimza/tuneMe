import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb

tracks = pd.read_csv('tracks.csv')


tracks.dropna(inplace = True) # Removing all rows with null values 
tracks = tracks.drop(['id', 'id_artists'], axis = 1) # Removing columns id and id_artists

year = []
for tmp in tracks['release_date']:
  tmp2 = tmp[:4]
  year.append(int(tmp2))

tracks['year'] = year
tracks = tracks.drop(['release_date'], axis = 1)


tracks['year'] = tracks['year'].astype(str)

tracks['text'] = tracks['name'] + ' ' + tracks['artists'] + ' ' + tracks['year']
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(tracks['text'])

import pickle
file = open('model.pkl', 'rb')
model = pickle.load(file)

feature_names = ['duration_ms', 'explicit','acousticness', 'danceability', 'energy', 'key', 'loudness','instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence','mode']
train_cols = feature_names

def recommend_songs(song_name, artist, year, tracks=tracks):
    # Find similar songs
    query = tfidf.transform([song_name + ' ' + artist + ' ' + str(year)])
    similarities = cosine_similarity(X, query).flatten()
    similar_indices = similarities.argsort()[::-1][1:11]
    similar_songs = tracks.iloc[similar_indices]

    # Predict popularity of similar songs
    dtest = xgb.DMatrix(similar_songs[train_cols])
    similar_songs['popularity_score'] = model.predict(dtest)

    # Sort by popularity and return top 5 recommendations
    recommendations = similar_songs.sort_values('popularity_score', ascending=False).head(5)
    return recommendations