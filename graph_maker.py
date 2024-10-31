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




# STEPS NOT BATCH NO.


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

data = np.array(data, dtype=float)
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
data = np.array(data, dtype=float)
sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]
xax = sorted_data[:, 0]
yax = sorted_data[:, 1]

spl = interpolate.CubicSpline(xax, yax)
xnew = np.linspace(0.0, 1.0, num=1001)
plt.plot(xnew, spl(xnew))
plt.savefig(os.path.join(script_dir, "code", "output", "noise_red_vs_avg_SNR_fac.png"))
plt.clf()






with open(os.path.join(script_dir, "code", "output", "noise_shot_vs_SNR_dB.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
shot_x = data[:, 0]
shot_y = data[:, 1]

with open(os.path.join(script_dir, "code", "output", "noise_white_vs_SNR_dB.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
white_x = data[:, 0]
white_y = data[:, 1]

with open(os.path.join(script_dir, "code", "output", "noise_pink_vs_SNR_dB.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
pink_x = data[:, 0]
pink_y = data[:, 1]

plt.scatter(shot_x, shot_y, color='blue', label='Shot Noise', s=0.3, alpha=0.5)
plt.scatter(pink_x, pink_y, color='red', label='Pink Noise', s=0.3, alpha=0.5)
plt.scatter(white_x, white_y, color='green', label='White Noise', s=0.3, alpha=0.5)
plt.xlabel('SNR in dB')
plt.ylabel('Noise reduction in dB')
#plt.title('Scatter Plot')
plt.legend()
plt.savefig(os.path.join(script_dir, "code", "output", "noise_reduction_by_type.png"))
plt.clf()








zip_file_path = os.path.join(script_dir, "code", "output")
folder_path = os.path.join(script_dir, "code", "output")
shutil.make_archive(zip_file_path, 'zip', folder_path)
print("Zipped all outputs")
print("Done!")