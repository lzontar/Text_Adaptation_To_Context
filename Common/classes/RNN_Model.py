#  https://github.com/joydeb28/NLP-Notebooks/blob/master/1.1-language-model/language_model_keras.ipynb
import json
import os
import random
import pandas as pd

from numba import jit, cuda, vectorize
from keras import backend as K

import keras
from numpy import array
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import load_model


class Preprocessing:
    _data_absolute_path = 'C:/Luka/School/Bachelor/Bachelor\'s thesis/Text_Adaptation/Data/covid-19/'

    def __init__(self):
        self.data = None
        self.vocab_size = None
        self.encoded_data = None
        self.max_length = None
        self.sequences = None
        self.x = None
        self.y = None
        self.tokenizer = None

    def load_data(self):
        self.data = []
        self.data = self.data + list(pd.read_csv(self._data_absolute_path + 'official_statements/data.csv')['content'].iloc[:5])
        self.data = self.data + list(pd.read_csv(self._data_absolute_path + 'news/news.csv')['text'].iloc[:5])
        self.data = self.data + list(pd.read_csv(self._data_absolute_path + 'tweets/covid19_tweets.csv')['text'].iloc[:5])

        rootdir = self._data_absolute_path + 'research_articles/document_parses/pdf_json'
        dataset = []
        for subdir, dirs, files in os.walk(rootdir):
            ix = 0
            for f in files:
                if ix == 1:
                    break
                file = open(self._data_absolute_path + 'research_articles/document_parses/pdf_json/' + f, 'r')
                json_file = json.loads(str(file.read()))
                research_article = ''
                for paragraph in json_file['body_text']:
                    research_article = research_article + paragraph['text']

                file.close()
                self.data.append(research_article)
                ix = ix + 1
        self.data = list(filter(lambda x: type(x) is str, self.data))

    def load_data_fine_tune_soc_med(self):
        self.data = []
        self.data = self.data + list(pd.read_csv(self._data_absolute_path + 'tweets/covid19_tweets.csv')['text'].iloc[:200])
        self.data = list(filter(lambda x: type(x) is str, self.data))

    def encode_data(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.data)
        self.encoded_data = self.tokenizer.texts_to_sequences(self.data)
        print(self.encoded_data)
        self.vocab_size = len(self.tokenizer.word_counts) + 1

    def generate_sequence(self):
        seq_list = list()
        for item in self.encoded_data:
            l = len(item)
            for id in range(1, l):
                seq_list.append(item[:id + 1])
        self.max_length = max([len(seq) for seq in seq_list])
        self.sequences = pad_sequences(seq_list, maxlen=self.max_length, padding='pre')
        print(self.sequences)
        self.sequences = array(self.sequences)

    def get_data(self):
        self.x = self.sequences[:, :-1]
        self.y = self.sequences[:, -1]
        print("y before:", self.y)
        self.y = to_categorical(self.y, num_classes=self.vocab_size)
        print("y After:", self.y)


class Model:
    def __init__(self, params):
        self.model = None
        self.history = None
        self.x = None
        self.y = None
        self.vocab_size = params['vocab_size']
        self.max_len = params['max_len']
        self.activation = params['activation']
        self.optimizer = params['optimizer']
        self.epochs = params['epochs']
        self.metrics = params['metrics']

    def create_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, 10, input_length=self.max_len - 1))
        self.model.add(LSTM(50))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(self.vocab_size, activation=self.activation))
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=self.metrics)
        print(self.model.summary())

    def run(self):
        self.history = self.model.fit(self.x, self.y, epochs=self.epochs)

    def save(self, path="./Data/Model/lang_model-covid.h5"):
        self.model.save(path)

    def fine_tune(self, ending):
        self.model = load_model("./Data/Model/lang_model-covid.h5")
        self.model.summary()
        self.model.trainable = False
        input_tensor = self.model.output
        input_tensor = Embedding(self.vocab_size, 10, input_length=self.max_len - 1, name='embed_new_1')(input_tensor)
        input_tensor = LSTM(50, name='lstm_new_1')(input_tensor)
        input_tensor = Dropout(0.1, name='dropout_new_1')(input_tensor)
        outputs = Dense(self.vocab_size, activation=self.activation, name='dense_new_1')(input_tensor)

        self.model = tf.keras.Model(inputs=self.model.input, outputs=outputs)
        self.model.summary()

        for layer in self.model.layers[:4]:
            print(layer, layer.trainable)
            layer.trainable = False

        self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(1e-5), metrics=self.metrics)

        self.history = self.model.fit(self.x, self.y, epochs=self.epochs)
        self.model.summary()
        self.save("./Data/Model/lang_model-covid-" + ending + ".h5")


class Prediction():
    def __init__(self, tokenizer, max_len):
        self.model = None
        self.tokenizer = tokenizer
        self.idx2word = {v: k for k, v in self.tokenizer.word_index.items()}
        self.max_length = max_len

    def load_model(self, ending=None):
        self.model = load_model("./Data/Model/lang_model-covid" + ("" if ending is None else ("-" + ending)) + ".h5")

    def predict_sequence(self, text, num_words):
        for id in range(num_words):
            encoded_data = self.tokenizer.texts_to_sequences([text])[0]
            padded_data = pad_sequences([encoded_data], maxlen=self.max_length - 1, padding='pre')
            y_pred = self.model.predict(padded_data)
            y_pred = np.argmax(y_pred)
            predict_word = self.idx2word[y_pred]
            text += ' ' + predict_word
        return text
