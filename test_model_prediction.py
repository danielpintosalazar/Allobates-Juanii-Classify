#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:20:45 2024

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

def build_predictions(audio_dir, labels, config):
    """
    Build predictions for audio files in a directory.

    Args:
    - audio_dir: Directory containing audio files.
    - labels: List of unique class labels.
    - config: Configuration object.

    Returns:
    - y_pred: Predicted class labels.
    - fn_prob: Dictionary mapping filenames to predicted probabilities.
    """
    y_pred = []
    fn_prob = {}

    for fn in tqdm(os.listdir(audio_dir)):
        rate, signal = wavfile.read(os.path.join(audio_dir, fn))
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
        
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()
    
    return y_pred, fn_prob

CONFIG_MODE = 'conv'
AUDIO_DIR = 'test_clean'
TRAIN_CSV = os.path.join('summaries', 'summary.csv')
TEST_CSV = os.path.join('summaries', 'Tests Table Selections.csv')
PICKLE_PATH = os.path.join('pickles', f'{CONFIG_MODE}.p')
PREDICTION_CSV = os.path.join('summaries', f'test_model_{CONFIG_MODE}_predictions.csv')

# Read the training and testing CSV files into DataFrames
df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)

# Extract unique labels
labels = list(np.unique(df_train['label']))

# Load the configuration object from the pickle file
with open(PICKLE_PATH, 'rb') as handle:
    config = pickle.load(handle)

# Load the pre-trained model
model = load_model(config.model_path)

# Build predictions for audio files
y_pred, fn_prob = build_predictions(AUDIO_DIR, labels, config)

# Store predicted probabilities in a DataFrame
y_probs = []
for i, row in df_test.iterrows():
    y_prob = fn_prob[row['Record Name']]
    y_probs.append(y_prob)
    for c, p in zip(labels, y_prob):
        df_test.at[i, c] = p

# Assign predicted labels to the DataFrame
y_pred = [labels[np.argmax(y)] for y in y_probs]
df_test['y_pred'] = y_pred

# Save the DataFrame with predictions to a CSV file
df_test.to_csv(PREDICTION_CSV, index=False)