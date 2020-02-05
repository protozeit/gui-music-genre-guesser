from tqdm import tqdm
from keras.utils import np_utils

import keras
import librosa
import multiprocessing
import numpy as np
import os
import pandas as pd
import sklearn as skl

from utils import load_model_from_file, extract_features, inv_genre_dict

def predict(X, modelname):
    model = load_model_from_file(modelname)

    print(model.predict(X))
    print(inv_genre_dict[model.predict_classes(X)[0]])

def main():
    print('input filename of wav: ')
    s = input()

    predict(extract_features(s), 'models/DNN.h5')

if __name__ == '__main__':
    main()

