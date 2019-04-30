from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pickle

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras

import numpy as np

# https://stackoverflow.com/questions/45735070/keras-text-preprocessing-saving-tokenizer-object-to-file-for-scoring
# https://www.tensorflow.org/alpha/tutorials/text/text_classification_rnn

class ImdbMlService():

    model = None
    train_data = None
    test_data = None
    dataset = None
    test_dataset = None
    train_dataset = None
    info = None
    tokenizer = None
    MODEL_FILE_PATH = './models/'
    MODEL_FILE_NAME = 'simple_text_imdb.h5'

    def __init__(self):
        try:
            print("ImdbMlService")
            np.set_printoptions(precision=6, suppress=True)
            self.get_model()
        except Exception as ex:
            raise ex

    def download_data(self):
        print("download_data()")
        # download data
        self.dataset, self.info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
        self.train_dataset, self.test_dataset = self.dataset['train'], self.dataset['test']
        self.tokenizer = self.info.features['text'].encoder
        #BUFFER_SIZE = 10000
        #BUFFER_SIZE = 6000
        #train_dataset = train_dataset.shuffle(BUFFER_SIZE)
        #batch_size = 64
        batch_size = 24
        self.train_dataset = self.train_dataset.padded_batch(
            batch_size, self.train_dataset.output_shapes)
        self.test_dataset = self.test_dataset.padded_batch(
            batch_size, self.test_dataset.output_shapes)

    def pad_to_size(self, vec, size):
        print("pad_to_size()")
        zeros = [0] * (size - len(vec))
        vec.extend(zeros)
        return vec

    def sample_predict(self, sentence, pad):
        print("simple_predict()")
        tokenized_sample_pred_text = self.tokenizer.encode(sentence)
        #print(tokenized_sample_pred_text)
        #model.summary()
        if pad:
            tokenized_sample_pred_text = self.pad_to_size(tokenized_sample_pred_text, 64)
        predictions = self.model.predict(tf.expand_dims(tokenized_sample_pred_text, 0))
        return predictions

    def create_model(self, train_data, test_data):
        print("create_model(...)")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.tokenizer.vocab_size, 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        history = model.fit(self.train_dataset, epochs=10, validation_data=self.test_dataset)
        #plot_graphs(history, 'accuracy')
        return (model, train_data, test_data)


    def save_model(self):
        print("save_model")
        file_name = self.MODEL_FILE_PATH + self.MODEL_FILE_NAME
        # saving
        with open(file_name +"_wi.pickle", 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.model.save(file_name)

    def load_model(self):
        print("load_model()")
        file_name = self.MODEL_FILE_PATH + self.MODEL_FILE_NAME
        # loading
        with open(file_name +"_wi.pickle", 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        new_model = keras.models.load_model(file_name)
        return new_model

    def train_save_model(self):
        print("train_save__model()")
        # Create a basic model instance
        self.download_data()
        (self.model, self.train_data, self.test_data) = self.create_model(
            self.train_data, self.test_data)
        self.model.summary()
        self.save_model()

    def get_model(self):
        print("get_model()")
        file_name = self.MODEL_FILE_PATH + self.MODEL_FILE_NAME
        print("Model file: " + file_name)
        if not os.path.exists(file_name):
            print("Model File doesn't exist")
        else:
            if self.model is None:
                self.model = self.load_model()
        return self.model

    def evaluate_text(self, text):
        print("evaluate_text(text)")
        if self.model is None:
            self.model = self.get_model()
        return self.sample_predict(text, True)
