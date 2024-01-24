#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 11:39:56 2024

@author: asus-pc
"""
import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt

def plot_signal(signal):
    fig, ax = plt.subplots()
    fig.suptitle('Time Series', size=16)
    ax.plot(signal)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def plot_fft(fft):
    fig, ax = plt.subplots()
    fig.suptitle('Fourier Transform', size=16)
    Y, freq = fft[0], fft[1]
    ax.plot(freq, Y)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def plot_fbank(fbank):
    fig, ax = plt.subplots()
    fig.suptitle('Filter Bank Coefficients', size=16)
    ax.imshow(fbank, cmap='hot', interpolation='nearest')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def plot_mfccs(mfccs):
    fig, ax = plt.subplots()
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    ax.imshow(mfccs, cmap='hot', interpolation='nearest')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)

def add_attrs(csv_path, audio_dir, col_record, label=None):
    """
    Adds attributes to a DataFrame based on information from a CSV file and audio files.

    Parameters:
        csv_path (str): Path to the CSV file.
        audio_dir (str): Directory containing the audio files.
        col_record (str): Column in CSV containing the record names.
        label (int): Label to classify records.

    Returns:
        pd.DataFrame: DataFrame with added attributes.
    """
    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Iterate through rows in the DataFrame
    for i in df.index:
        # Add path to record
        df.at[i, 'path'] = os.path.join(audio_dir, df.at[i, col_record])
        
        # Add length, rate attributes with rate and signal of the record
        rate, signal = wavfile.read(df.at[i, 'path'])
        df.at[i, 'rate'] = rate
        df.at[i, 'length'] = signal.shape[0] / rate
        
        # Add label to classify records
        if label == 0 or label == 1:
            df['label'] = label
    
    return df