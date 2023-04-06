import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings('ignore')

tracks = pd.read_csv('tracks.csv')
tracks.head()

print(tracks.info())

tracks.shape

tracks.dropna(inplace = True)

tracks = tracks.drop(['id', 'id_artists'], axis = 1)

tracks.shape


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import xgboost as xgb

tracks['release_date']
# tracks['release_date']
year = []
for tmp in tracks['release_date']:
  tmp2 = tmp[:4]
  year.append(int(tmp2))

tracks['year'] = year
tracks = tracks.drop(['release_date'], axis = 1)
tracks.info()

tracks['year'] = tracks['year'].astype(str)

tracks['text'] = tracks['name'] + ' ' + tracks['artists'] + ' ' + tracks['year']
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(tracks['text'])

from yellowbrick.target import FeatureCorrelation
feature_names = ['duration_ms', 'explicit','acousticness', 'danceability', 'energy', 'key', 'loudness','instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence','mode']
A, b = tracks[feature_names], tracks['popularity']

# Create a list of the feature names
features = np.array(feature_names)

# Instantiate the visualizer
visualizer = FeatureCorrelation(labels=features)

plt.rcParams['figure.figsize']=(20,20)
visualizer.fit(A, b)     # Fit the data to the visualizer
visualizer.show()

sorted_tracks = tracks.sort_values(by=['popularity'], ascending=False).head(1000)
sorted_tracks.drop_duplicates(subset=['name'], keep='first', inplace=True)
sorted_tracks.head(5)

train_cols = feature_names
dtrain = xgb.DMatrix(tracks[train_cols], label=tracks['popularity'])
params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
model = xgb.train(params, dtrain)

import pickle
with open('model.pkl', 'wb') as file:
  pickle.dump(model, file)

tracks.info()

from yellowbrick.target import FeatureCorrelation

feature_names = ['duration_ms', 'explicit','acousticness', 'danceability', 'energy', 'key', 'loudness','instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence','mode']

X, y = tracks[feature_names], tracks['popularity']

# Create a list of the feature names
features = np.array(feature_names)

# Instantiate the visualizer
visualizer = FeatureCorrelation(labels=features)

plt.rcParams['figure.figsize']=(20,20)
visualizer.fit(X, y)     # Fit the data to the visualizer
visualizer.show()

tracks['release_date']

# tracks['release_date']
year = []
for tmp in tracks['release_date']:
  tmp2 = tmp[:4]
  year.append(int(tmp2))

tracks['year'] = year

tracks = tracks.drop(['release_date'], axis = 1)
tracks.info()

tracks['name'].nunique(), tracks.shape

tracks = tracks.sort_values(by=['popularity'], ascending=False)
tracks.drop_duplicates(subset=['name'], keep='first', inplace=True)

"""Here we are drawing the distribution plot of float values to get distribution insight of the features. First let's find the number of float features available, further we will draw the distribution graph."""

dist = []
for col in tracks.columns:
  if tracks[col].dtype == 'float':
	  dist.append(col)

len(dist)

plt.subplots(figsize = (15, 5))
for i, col in enumerate(dist):
  plt.subplot(5, 5, i + 1)
  sns.distplot(tracks[col])
plt.tight_layout()
plt.show()

tracks = tracks.sort_values(by=['popularity'], ascending=False).head(10000)
tracks.head(5)

"""As the previous dataset didn't have genres in it, here I am uploading the artists.csv file which is also present in kaggle 'Spotify Dataset'.

*https://www.kaggle.com/datasets/lehaknarnauli/spotify-datasets?resource=download*
"""

data = pd.read_csv("artists.csv")
data.head(5)

data = data.drop(['id'], axis = 1)
data.head()

"""Again we are dropping the null values over here, which may create a noise in training our model."""

data.isnull().sum()

data.dropna(inplace = True)
data.shape

data = data.sort_values(by=['popularity'], ascending = False)
data.head(5)

# Commented out IPython magic to ensure Python compatibility.
# %%capture
song_vectorizer = CountVectorizer()
song_vectorizer.fit(data['genres'])

"""Here we will define a similarity function which will define the similarities between the input song and all other songs from the dataset, it takes genres to do so, we can also use different field like danceability or acousticness to compute the similarness."""

from sklearn.metrics.pairwise import cosine_similarity

def get_similarities(song_name, data):
    # Filter data for input song and current song
    input_song_text = song_vectorizer.transform(data[data['name']==song_name]['genres']).toarray()
    input_song_num = data[data['name']==song_name].select_dtypes(include=np.number).to_numpy()[0]
    data_text = song_vectorizer.transform(data['genres']).toarray()
    data_num = data.select_dtypes(include=np.number).to_numpy()

    # Calculate similarities for text and numeric features using numpy arrays
    text_sim = cosine_similarity(input_song_text, data_text).ravel()
    num_sim = cosine_similarity(input_song_num.reshape(1,-1), data_num).ravel()
    sim = text_sim + num_sim

    # Return list of similarities for each row of the dataset
    return sim.tolist()

"""Now we will define a recommend function which user can use directly to see the recommended options accoridng to our similarity function."""

def recommend_songs(song_name, data=data):
  # Base case
  if tracks[tracks['name'] == song_name].shape[0] == 0:
    print('This song is either not so popular or you\
    have entered invalid_name.\n Some songs you may like:\n')

    for song in data.sample(n=5)['name'].values:
      print(song)
    return

  data['similarity_factor'] = get_similarities(song_name, data)

  data.sort_values(by=['similarity_factor', 'popularity'],
				           ascending = [False, False],
				           inplace=True)
  pd.display(data[['name']][2:7])
  # First song will be the input song itself as the similarity will be highest.

recommend_songs('Dandelions')

"""Above you can see the custom recommendation song accoridng to the given input song."""

# import pickle
# with open('my_model.pkl', 'wb') as file:
#   pickle.dump(recommend_songs, file)

