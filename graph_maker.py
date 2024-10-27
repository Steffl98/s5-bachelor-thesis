import struct
import math
import random
import os
import sys
import time
import numpy as np
from typing import Tuple, Optional, Literal
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import pandas as pd
import shutil
Initialization = Literal['dense_columns', 'dense', 'factorized']
import csv
from scipy import interpolate



script_dir = os.path.join("C:\\", "Users", "stefa", "OneDrive", "Desktop", "Uni", "Bachelorarbeit")
try:
    argument1 = sys.argv[1]
    print("Using working directory: ", argument1)
    script_dir = argument1
except IndexError:
    print("Usage: python main.py <audio_dir_path>")
    print("Attempting to use default path...")



# Read the CSV file
with open(os.path.join(script_dir, "code", "output", "dataset_spectrogram.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

# Convert the list to a NumPy array
data = np.array(data, dtype=float)  # Adjust the dtype if necessary

sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]

freq_axis = sorted_data[:, 0]
yax = sorted_data[:, 1]

plt.plot(freq_axis, yax, color='blue', label='Before augmentations')
plt.title("Data Set Audio Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
#plt.xlim(0, 2000)
plt.xscale('log', base=10)
plt.xlim(20, 8000)
plt.ylim(0, 20000)
plt.savefig(os.path.join(script_dir, "code", "output", "data_set_spectrum.png"))
plt.clf()








with open(os.path.join(script_dir, "code", "output", "noise_red_vs_avg_SNR_fac.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

# Convert the list to a NumPy array
data = np.array(data, dtype=float)  # Adjust the dtype if necessary

sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]

xax = sorted_data[:, 0]
yax = sorted_data[:, 1]

spl = interpolate.QuadraticSpline(xax, yax)
xnew = np.linspace(0.0, 1.0, num=1001)
plt.plot(xnew, spl(xnew))
plt.savefig(os.path.join(script_dir, "code", "output", "aaa.png"))