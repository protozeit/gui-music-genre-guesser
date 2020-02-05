from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop, Nadam

from tqdm import tqdm
from keras.utils import np_utils

import pickle
import keras
import librosa
import multiprocessing
import numpy as np
import os
import pandas as pd
import sklearn as skl

def predict(X, modelname):

    model = Sequential()
    model.add(Dense(128, input_dim=207, activation='relu'))

    model.add(Dropout(0.6))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(8, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    model.load_weights(modelname)

    print(model.predict(X))
    print(inv_genre_dict[model.predict_classes(X)[0]])

def extract_features(wavfile):
    pca = pickle.load(open('pca', 'rb'))

    X = compute_features(wavfile).to_frame()
    X = skl.preprocessing.StandardScaler().fit_transform(X)

    X = pca.transform(X)
    print('extracted features of form : '+ str(X.shape))
    return X

def main():
    print('input filename of wav: ')
    s = input()

    predict(extract_features(s), 'models/DNN.h5')

if __name__ == '__main__':
    main()

