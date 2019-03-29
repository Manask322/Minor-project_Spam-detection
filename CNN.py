import WordEmbeddings

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input, LSTM
import numpy as np
import math
import random
import sys
import gensim
import warnings
import re
from itertools import chain
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
 
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import wordnet
from nltk.corpus import stopwords


def main():
    words = ['go', 'jurong', 'point', 'crazy', 'available', 'bugis', 'n', 'great', 'world', 'la', 'e', 'buffet', 'cine', 'got', 'amore', 'wat']
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=600000)
    input = []
    for w in words :
        if w in model.vocab  :
            input.append(model[w])
        else:
            input.append(model[random.choice(model.index2entity)])
    print(input[0])
    model = Sequential()
    model.add(Conv1D(32,4,activation='relu',input_shape=(1200,1)))
    model.add(LSTM(100, activation='sigmoid', recurrent_activation='hard_sigmoid', use_bias=True, dropout=0.1, recurrent_dropout=0.0))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    x = input
    x = np.reshape(x,[4,1200,1])
    y = np.array([1,0,0,1],dtype='f')
    model.summary()
    # dataset= WordEmbeddings.load_file() 
    # dataset= WordEmbeddings.get_vectors(dataset)
    # print(len(dataset))
    # #len(dataset) = 2274
    # #dim = 300
    # k=dataset[0]
    # print(k.vectors)
    # print(k.label,k.words,len(k.vectors),len(k.vectors[0]))
    # x_train = dataset

if __name__ == "__main__":
    main()