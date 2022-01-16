#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 15:26:55 2022

@author: abdul
"""

import numpy as np
import pandas as pd
import os
import warnings
import tensorflow as tf

from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model
warnings.filterwarnings('ignore')

from keras.models import Model
from keras.layers import Reshape
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Dropout
from keras.layers import Add, Activation, Lambda
import matplotlib.pyplot as plt



#LOADING DATA
header = ['user_id','item_id','rating','timestamp']
dataset = pd.read_csv('u.data',sep = '\t',names = header)
dataset.head()

dataset = dataset.drop(['timestamp'], axis=1)



# DATA PREPROCESSING
user_ids = dataset["user_id"].unique().tolist()

user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}


movie_ids = dataset["item_id"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}


dataset["user"] = dataset["user_id"].map(user2user_encoded)
dataset["movie"] = dataset["item_id"].map(movie2movie_encoded)

num_users = len(user2user_encoded)
num_movies = len(movie_encoded2movie)


dataset["rating"] = dataset["rating"].values.astype(np.float32)
# min and max ratings will be used to normalize the ratings later
min_rating = min(dataset["rating"])
max_rating = max(dataset["rating"])

print("Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}".format(
        num_users, num_movies, min_rating, max_rating))


#SPLITTING THE DATASET

dataset = dataset.sample(frac=1, random_state=42)
x = dataset[["user", "movie"]].values
# Normalize the targets between 0 and 1. Makes it easy to train.
y = dataset["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
# Assuming training on 90% of the data and validating on 10%.
train_indices = int(0.9 * dataset.shape[0])
x_train, x_val, y_train, y_val = (x[:train_indices], x[train_indices:],
                                  y[:train_indices], y[train_indices:])

x_train.shape, x_val.shape, y_train.shape, y_val.shape

X_train_array = [x_train[:, 0], x_train[:, 1]]
X_val_array = [x_val[:, 0], x_val[:, 1]]



#MODELING
EMBEDDING_SIZE = 50

user = Input(shape=(1,))
u = Embedding(num_users, EMBEDDING_SIZE, embeddings_initializer='he_normal',
              embeddings_regularizer=l2(1e-6))(user)

u = Reshape((EMBEDDING_SIZE,))(u)

movie = Input(shape=(1,))
m = Embedding(num_movies, EMBEDDING_SIZE, embeddings_initializer='he_normal',
              embeddings_regularizer=l2(1e-6))(movie)
m = Reshape((EMBEDDING_SIZE,))(m)

x = Dot(axes=1)([u, m])
x = Activation('sigmoid')(x)
x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)

model = Model(inputs=[user, movie], outputs=x)
opt = Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=opt)

model.summary()

history = model.fit(x = X_train_array, y = y_train, batch_size=64, epochs=10, verbose=1, 
                    validation_data=(X_val_array, y_val))

#PLOTTING TRAINING AND VALIDATION LOSS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper right')
plt.show()


y_pred = model.predict(X_val_array)


plt.scatter(y_val, y_pred)
plt.title('Predictions')
plt.show()












