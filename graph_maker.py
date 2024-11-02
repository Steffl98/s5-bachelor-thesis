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

with open(os.path.join(script_dir, "code", "output", "target_spectrum.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

data = np.array(data, dtype=float)
sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices] / 5.0
freq_axis = sorted_data[:, 0]
yax = sorted_data[:, 1]
plt.plot(freq_axis, yax, color='orange', label='After augmentations')

plt.savefig(os.path.join(script_dir, "code", "output", "augmentations_before_after.png"))
plt.clf()








# Read the CSV file
with open(os.path.join(script_dir, "code", "output", "target_spectrum.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

data = np.array(data, dtype=float)
sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]
freq_axis = sorted_data[:, 0]
yax = sorted_data[:, 1]
plt.plot(freq_axis, yax, color='blue', label='Target')
plt.title("Data Set Audio Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
#plt.xlim(0, 2000)
plt.xscale('log', base=10)
plt.xlim(20, 8000)
plt.ylim(0, 100000)

with open(os.path.join(script_dir, "code", "output", "output_spectrum.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

data = np.array(data, dtype=float)
sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]
freq_axis = sorted_data[:, 0]
yax = sorted_data[:, 1]
plt.plot(freq_axis, yax, color='orange', label='Output')

plt.savefig(os.path.join(script_dir, "code", "output", "target_vs_output.png"))
plt.clf()









with open(os.path.join(script_dir, "code", "output", "noise_red_vs_SNR_fac.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
xdata = data[:, 0]
ydata = data[:, 1]
plt.scatter(xdata, ydata, color='orange', label='Noise reduction', s=0.3)
with open(os.path.join(script_dir, "code", "output", "noise_red_vs_avg_SNR_fac.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]
xax = sorted_data[:, 0]
yax = sorted_data[:, 1]

spl = interpolate.CubicSpline(xax, yax)
xnew = np.linspace(0.05, 0.95, num=1001)
plt.plot(xnew, spl(xnew))
plt.savefig(os.path.join(script_dir, "code", "output", "noise_red_vs_avg_SNR_fac.png"))
plt.clf()






with open(os.path.join(script_dir, "code", "output", "noise_shot_vs_SNR_dB.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
shot_x = data[:, 0]
shot_y = data[:, 1]
indices = np.random.permutation(len(shot_x))
shot_x = shot_x[indices]
shot_y = shot_y[indices]

with open(os.path.join(script_dir, "code", "output", "noise_white_vs_SNR_dB.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
white_x = data[:, 0]
white_y = data[:, 1]
indices = np.random.permutation(len(white_x))
white_x = white_x[indices]
white_y = white_y[indices]

with open(os.path.join(script_dir, "code", "output", "noise_pink_vs_SNR_dB.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
pink_x = data[:, 0]
pink_y = data[:, 1]
indices = np.random.permutation(len(pink_x))
pink_x = pink_x[indices]
pink_y = pink_y[indices]

max_len = max(len(shot_x), len(pink_x), len(white_x))

for i in range(max_len):
    val_pink_x = pink_x[i] if i < len(pink_x) else None
    val_white_x = white_x[i] if i < len(white_x) else None
    val_shot_x = shot_x[i] if i < len(shot_x) else None
    val_pink_y = pink_y[i] if i < len(pink_y) else None
    val_white_y = white_y[i] if i < len(white_y) else None
    val_shot_y = shot_y[i] if i < len(shot_y) else None

    if i < len(pink_x):
        plt.scatter(val_pink_x, val_pink_y, color='blue', label='Pink Noise' if i == 0 else "", s=0.15, alpha=0.5)
    if i < len(white_x):
        plt.scatter(val_white_x, val_white_y, color='green', label='White Noise' if i == 0 else "", s=0.15, alpha=0.5)
    if i < len(shot_x):
        plt.scatter(val_shot_x, val_shot_y, color='red', label='Shot Noise' if i == 0 else "", s=0.15, alpha=0.5)

#plt.scatter(shot_x, shot_y, color='blue', label='Shot Noise', s=0.3, alpha=0.5)
#plt.scatter(pink_x, pink_y, color='red', label='Pink Noise', s=0.3, alpha=0.5)
#plt.scatter(white_x, white_y, color='green', label='White Noise', s=0.3, alpha=0.5)
plt.xlabel('SNR in dB')
plt.ylabel('Noise reduction in dB')
#plt.title('Scatter Plot')
plt.legend()
plt.savefig(os.path.join(script_dir, "code", "output", "noise_reduction_by_type.png"), dpi=600)
plt.clf()








with open(os.path.join(script_dir, "code", "output", "test_performance.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
test_data = np.array(data, dtype=float)

with open(os.path.join(script_dir, "code", "output", "test_l1_performance.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
test_l1_data = np.array(data, dtype=float)

with open(os.path.join(script_dir, "code", "output", "training_loss.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
loss_data = np.array(data, dtype=float)

x_data_1 = test_data[:, 0]
y_data_1 = test_data[:, 1]
x_data_2 = loss_data[:, 0]
y_data_2 = loss_data[:, 1]

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('X data 1')
ax1.set_ylabel('Y data 1', color=color)
ax1.plot(x_data_1, y_data_1, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Y data 2', color=color)
ax2.plot(x_data_2, y_data_2, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
plt.savefig(os.path.join(script_dir, "code", "output", "training_vs_test_loss.png"), dpi=600)
plt.clf()

















zip_file_path = os.path.join(script_dir, "code", "output")
folder_path = os.path.join(script_dir, "code", "output")
shutil.make_archive(zip_file_path, 'zip', folder_path)
print("Zipped all outputs")
print("Done!")