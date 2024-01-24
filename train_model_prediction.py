#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:36:01 2024

@author: asus-pc
"""

import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
from sklearn.metrics import accuracy_score

def build_predictions(audio_dir, fn2class, labels, config):
    """
    Build predictions for audio files in a directory.

    Args:
    - audio_dir: Directory containing audio files.
    - fn2class: Dictionary mapping filenames to class labels.
    - labels: List of unique class labels.
    - config: Configuration object.

    Returns:
    - y_true: True class labels.
    - y_pred: Predicted class labels.
    - fn_prob: Dictionary mapping filenames to predicted probabilities.
    """
    y_true = []
    y_pred = []
    fn_prob = {}

    for fn in tqdm(os.listdir(audio_dir)):
        rate, signal = wavfile.read(os.path.join(audio_dir, fn))
        label = fn2class[fn]
        c = labels.index(label)
        y_prob = []

        for i in range(0, signal.shape[0] - config.step, config.step):
            sample = signal[i:i+config.step]
            x = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
            x = (x - config.min) / (config.max - config.min)

            if config.mode == 'conv':
                x = x.reshape(1, x.shape[0], x.shape[1], 1)
            elif config.mode == 'time':
                x = np.expand_dims(x, axis=0)

            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(c)
        
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()
    
    return y_true, y_pred, fn_prob


CONFIG_MODE = 'conv'
AUDIO_DIR = 'train_clean'
TRAIN_CSV = os.path.join('summaries', 'summary.csv')
PICKLE_PATH = os.path.join('pickles', f'{CONFIG_MODE}.p')
PREDICTION_CSV = os.path.join('summaries', 'train_model_predictions.csv')

# Read the training CSV file into a DataFrame
df = pd.read_csv(TRAIN_CSV)

# Extract unique labels and create a dictionary mapping filenames to class labels
labels = list(np.unique(df['label']))
fn2class = dict(zip(df['Record Name'], df['label']))

# Load the configuration object from the pickle file
with open(PICKLE_PATH, 'rb') as handle:
    config = pickle.load(handle)

# Load the pre-trained model
model = load_model(config.model_path)

# Build predictions for audio files
y_true, y_pred, fn_prob = build_predictions(AUDIO_DIR, fn2class, labels, config)

# Calculate accuracy score
acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)

# Store predicted probabilities in a DataFrame
y_probs = []
for i, row in df.iterrows():
    y_prob = fn_prob[row['Record Name']]
    y_probs.append(y_prob)
    for c, p in zip(labels, y_prob):
        df.at[i, c] = p

# Assign predicted labels to the DataFrame
y_pred = [labels[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred

# Save the DataFrame with predictions to a CSV file
df.to_csv(PREDICTION_CSV, index=False)
