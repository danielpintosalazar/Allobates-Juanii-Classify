#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:51:41 2024

@author: asus-pc
"""

import os
from utils import add_attrs
import pandas as pd

# Define constants
TRAIN_DIR = os.path.join('train_clips')
SUMMARY_CSV = os.path.join('summaries', 'summary.csv')
ALLOBATES_CSV = os.path.join('summaries', 'Allobates Juanii Table Selections.csv')
NOT_ALLOBATES_CSV = os.path.join('summaries', 'Not Allobates Juanii Table Selections.csv')

# Add attributes to positive and negative DataFrames
df_pos = add_attrs(ALLOBATES_CSV, TRAIN_DIR, 'Record Name', 1)
df_neg = add_attrs(NOT_ALLOBATES_CSV, TRAIN_DIR, 'Record Name', 0)

# Concatenate positive and negative DataFrames
df = pd.concat([df_pos, df_neg])

# Reset the index of the combined DataFrame
df.reset_index(drop=True, inplace=True)

# Save DataFrame to CSV
df.to_csv(SUMMARY_CSV, index=False)
