#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:47:13 2024

@author: asus-pc
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
from tqdm import tqdm
from python_speech_features import mfcc, logfbank
from utils import plot_signal, plot_fft, plot_fbank, plot_mfccs, envelope, calc_fft

def clean_audios(df, clean_dir):
    """
    Clean audio files by applying an envelope to remove silence and save the cleaned files.

    Parameters:
        df (pd.DataFrame): DataFrame containing audio file information.
        clean_dir (str): Directory to store the cleaned audio files.
    """
    if len(os.listdir(clean_dir)) > 0:
        for i in tqdm(df.index):
            record = df.iloc[i]
            os.remove(f'{clean_dir}/{record["Record Name"]}')
    
    if len(os.listdir(clean_dir)) == 0:
        for i in tqdm(df.index):
            record = df.iloc[i]
            path = record.path
            signal, rate = librosa.load(path, sr=16000)
            mask = envelope(signal, rate, 0.0005)
            wavfile.write(filename=f'{clean_dir}/{record["Record Name"]}', rate=rate, data=signal[mask])
            df.at[i, 'c_path'] = f'{clean_dir}/{record["Record Name"]}'

# Constants
SUMMARY_CSV = os.path.join('summaries', 'summary.csv')
CLEAN_DIR = os.path.join('train_clean')

# Read DataFrame from CSV
df = pd.read_csv(SUMMARY_CSV)

# Visualize labels distribution
labels = np.unique(df['label'])
labels_dist = df.groupby('label')['length'].mean()
fig, ax = plt.subplots()
ax.set_title('Labels Distribution')
ax.pie(labels_dist, labels=labels, startangle=90, autopct='%1.1f%%')
ax.axis('equal')
plt.show()

# Randomly select an index with label 1 and visualize various features
rdindex = np.random.choice(df[df['label'] == 1].index)
signal, rate = librosa.load(df.iloc[rdindex].path, sr=48000)
mask = envelope(signal, rate, 0.0005)
signal = signal[mask]
bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1200).T
mfccs = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1200).T
fft = calc_fft(signal, rate)

# Visualize features
plot_signal(signal)
plt.show()

plot_fft(fft)
plt.show()

plot_fbank(bank)
plt.show()

plot_mfccs(mfccs)
plt.show()

# Clean audio files and update DataFrame
clean_audios(df, CLEAN_DIR)

# Reset the index of the combined DataFrame
df.reset_index(drop=True, inplace=True)

# Save DataFrame to CSV
df.to_csv(SUMMARY_CSV, index=False)


