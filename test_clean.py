#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:21:39 2024

@author: asus-pc
"""

import os
from scipy.io import wavfile
import librosa
from tqdm import tqdm
from utils import add_attrs

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
            #mask = envelope(signal, rate, 0.0005)
            wavfile.write(filename=f'{clean_dir}/{record["Record Name"]}', rate=rate, data=signal)
            df.at[i, 'c_path'] = f'{clean_dir}/{record["Record Name"]}'

# Constants
TEST_CSV = os.path.join('summaries', 'Tests Table Selections.csv')
TEST_DIR = os.path.join('test_clips')
CLEAN_DIR = os.path.join('test_clean')

# Add attributes to CSV of records
df_test = add_attrs(TEST_CSV, TEST_DIR, 'Record Name')

# Clean audio files and update DataFrame
clean_audios(df_test, CLEAN_DIR)

# Save DataFrame to CSV
df_test.to_csv(TEST_CSV, index=False)


