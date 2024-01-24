#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:08:47 2024

@author: asus-pc
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from cfg import Config
from sklearn.utils.class_weight import compute_class_weight
from python_speech_features import mfcc
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
import pickle
from keras.callbacks import ModelCheckpoint

def build_rand_feat(df, n_samples, col_label, col_record, labels, label_dist, prob_dist, config):
    """
    Build random features and labels for model training.

    Parameters:
        df (pd.DataFrame): DataFrame containing audio file information.
        n_samples (int): Number of samples to generate.
        col_label (str): Column name containing labels.
        col_record (str): Column name containing record names.
        labels (list): List of unique labels.
        label_dist (pd.Series): Distribution of labels in the DataFrame.
        prob_dist (list): Probability distribution for random label selection.
        config (Config): Configuration object for audio processing.

    Returns:
        tuple: X (input features), y (labels).
    """
    X = []
    y = []
    
    _min, _max = float('inf'), -float('inf')
    
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(label_dist.index, p=prob_dist)
        f = np.random.choice(df[df[col_label] == rand_class].index)
        rate, signal = wavfile.read(df.iloc[f][col_record])
        label = df.at[f, col_label]
        rand_index = np.random.randint(0, signal.shape[0] - config.step)
        sample = signal[rand_index:rand_index+config.step]
        X_sample = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)

        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)

        X.append(X_sample)
        y.append(labels.index(label))
    
    config.min = _min
    config.max = _max

    X, y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    
    y = to_categorical(y, num_classes=2)
    config.data = (X, y)

    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return X, y


def get_conv_model(input_shape):
    """
    Build and return a convolutional neural network model.

    Parameters:
        input_shape (tuple): Shape of the input data.

    Returns:
        Sequential: Compiled convolutional neural network model.
    """
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.95))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def get_recurrent_model(input_shape):
    """
    Build and return a recurrent neural network model.

    Parameters:
        input_shape (tuple): Shape of the input data.

    Returns:
        Sequential: Compiled recurrent neural network model.
    """
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def generate_model(X, y, config_mode='conv'):
    """
    Generate a neural network model based on the configuration mode.

    Parameters:
        X (numpy.ndarray): Input features.
        y (numpy.ndarray): Labels.
        config_mode (str): Configuration mode, either 'conv' or 'time'.

    Returns:
        tuple: y_flat (flattened labels), model (generated neural network model).
    """
    y_flat = np.argmax(y, axis=1)
    if config_mode == 'conv':
        input_shape = (X.shape[1], X.shape[2], 1)
        model = get_conv_model(input_shape)
    elif config_mode == 'time':
        input_shape = (X.shape[1], X.shape[2])
        model = get_recurrent_model(input_shape)

    return y_flat, model


# Define the path to the training CSV file and create a configuration object
TRAIN_CSV = os.path.join('summaries', 'summary.csv')
config = Config(mode='conv')

# Read the training CSV file into a DataFrame
df = pd.read_csv(TRAIN_CSV)

# Extract unique labels and calculate label distribution
labels = list(np.unique(df['label']))
label_dist = df.groupby(['label'])['length'].mean()

# Calculate the number of samples for training
n_samples = 2 * int(df.length.sum() / 0.1)

# Calculate the probability distribution based on label distribution
prob_dist = label_dist / label_dist.sum()

# Build random features and labels for training
X, y = build_rand_feat(df, n_samples, 'label', 'c_path', labels, label_dist, prob_dist, config)

# Generate the model based on the selected mode in the configuration
y_flat, model = generate_model(X, y, config_mode=config.mode)

# Compute class weights for imbalanced classes
class_weight = compute_class_weight('balanced', classes=np.unique(y_flat), y=y_flat)
class_weight = dict(zip(np.unique(y_flat), class_weight))

# Define a ModelCheckpoint callback to save the best model during training
checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, period=1)

# Train the model with the provided data and settings
model.fit(X, y, epochs=1, batch_size=32, shuffle=True, class_weight=class_weight,
          validation_split=0.6, callbacks=[checkpoint])

# Save the trained model to the specified path
model.save(config.model_path)

