#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:16:23 2024

@author: asus-pc
"""

import os

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000, model_dir='models', pickle_path='pickles'):
        """
        Configuration class for audio processing and model training.

        Parameters:
            mode (str): Processing mode, either 'conv' or 'time'.
            nfilt (int): Number of filter banks.
            nfeat (int): Number of features.
            nfft (int): Size of the FFT.
            rate (int): Sampling rate of the audio.
            model_dir (str): Directory to store trained models.
            pickle_path (str): Directory to store pickled objects.
        """
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)
        self.model_path = os.path.join(model_dir, f'{mode}.model')
        self.p_path = os.path.join(pickle_path, f'{mode}.p')

