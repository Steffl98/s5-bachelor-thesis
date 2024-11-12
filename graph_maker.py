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



SNR_RANGE = 15.0


# STEPS NOT BATCH NO.


script_dir = os.path.join("C:\\", "Users", "stefa", "OneDrive", "Desktop", "Uni", "Bachelorarbeit")
try:
    argument1 = sys.argv[1]
    print("Using working directory: ", argument1)
    script_dir = argument1
except IndexError:
    print("Usage: python main.py <audio_dir_path>")
    print("Attempting to use default path...")


choice = input("dB mode? Type y for dB mode, else using snr fac by default.")

if (choice == "y"):
    USE_DB = True
else:
    USE_DB = False

# Read the CSV file
with open(os.path.join(script_dir, "code", "output", "dataset_spectrogram.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

data = np.array(data, dtype=float)
sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]
freq_axis = sorted_data[:, 0]
yax = sorted_data[:, 1]# * 4000.0
#yax = yax / np.mean(yax)
plt.plot(freq_axis, yax, color='blue', label='Before augmentations')

#plt.xlim(0, 2000)


with open(os.path.join(script_dir, "code", "output", "target_spectrum.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

data = np.array(data, dtype=float)
sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]
freq_axis = sorted_data[:, 0]
yax = sorted_data[:, 1]
#yax = yax / np.mean(yax)
plt.plot(freq_axis, yax, color='orange', label='After augmentations')
plt.title("Data Set Audio Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)

plt.xscale('log', base=10)
plt.xlim(20, 8000)
plt.legend()
#plt.ylim(0, 20000)

plt.savefig(os.path.join(script_dir, "code", "output", "augmentations_before_after.png"))
plt.clf()

















with open(os.path.join(script_dir, "code", "output", "dataset_spectrogram.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

data = np.array(data, dtype=float)
sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]
freq_axis = sorted_data[:, 0]
yax = sorted_data[:, 1]# * 4000.0
plt.plot(freq_axis, yax, color='blue', label='Data before augmentations')

with open(os.path.join(script_dir, "code", "output", "input_spectrum.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

data = np.array(data, dtype=float)
sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]
freq_axis = sorted_data[:, 0]
yax = sorted_data[:, 1]
plt.plot(freq_axis, yax, color='orange', label='Input dataset')
plt.title("Data Set Audio Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)

plt.xscale('log', base=10)
plt.xlim(20, 8000)
plt.legend()
plt.savefig(os.path.join(script_dir, "code", "output", "dataset_vs_input.png"))
plt.clf()













with open(os.path.join(script_dir, "code", "output", "input_spectrum.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

data = np.array(data, dtype=float)
sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]
freq_axis = sorted_data[:, 0]
yax1 = sorted_data[:, 1]# * 4000.0
#plt.plot(freq_axis, yax, color='blue', label='Data before augmentations')

with open(os.path.join(script_dir, "code", "output", "output_spectrum.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

data = np.array(data, dtype=float)
sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]
freq_axis = sorted_data[:, 0]
yax2 = sorted_data[:, 1]
plt.plot(freq_axis, yax2/yax1, color='orange', label='Transfer function')
plt.title("Data Set Audio Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)

plt.xscale('log', base=10)
plt.yscale('log', base=10)
plt.xlim(20, 8000)
plt.ylim(0.05, 20)
#plt.legend()
plt.savefig(os.path.join(script_dir, "code", "output", "transfer_function.png"))
plt.clf()
















with open(os.path.join(script_dir, "code", "output", "target_spectrum.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

data = np.array(data, dtype=float)
sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]
freq_axis = sorted_data[:, 0]
yax = sorted_data[:, 1]# * 4000.0
plt.plot(freq_axis, yax, color='blue', label='Target Data')

with open(os.path.join(script_dir, "code", "output", "input_spectrum.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

data = np.array(data, dtype=float)
sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]
freq_axis = sorted_data[:, 0]
yax = sorted_data[:, 1]
plt.plot(freq_axis, yax, color='orange', label='Input dataset')
plt.title("Data Set Audio Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)

plt.xscale('log', base=10)
plt.xlim(20, 8000)
plt.ylim(0, 0.0025)
plt.legend()
plt.savefig(os.path.join(script_dir, "code", "output", "target_vs_input.png"))
plt.clf()



















with open(os.path.join(script_dir, "code", "output", "input_spectrum.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

data = np.array(data, dtype=float)
sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]
freq_axis = sorted_data[:, 0]
yax = sorted_data[:, 1]
plt.plot(freq_axis, yax, color='orange', label='Input dataset')
plt.title("Data Set Audio Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)




with open(os.path.join(script_dir, "code", "output", "output_spectrum.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

data = np.array(data, dtype=float)
sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]
freq_axis = sorted_data[:, 0]
yax = sorted_data[:, 1]# * 4000.0
plt.plot(freq_axis, yax, color='green', label='Output')



with open(os.path.join(script_dir, "code", "output", "target_spectrum.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

data = np.array(data, dtype=float)
sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]
freq_axis = sorted_data[:, 0]
yax = sorted_data[:, 1]# * 4000.0
plt.plot(freq_axis, yax, color='blue', label='Target Data')


plt.xscale('log', base=10)
plt.xlim(20, 8000)
plt.ylim(0, 0.0025)
plt.legend()
plt.savefig(os.path.join(script_dir, "code", "output", "input_and_output_and_target.png"))
plt.clf()

























with open(os.path.join(script_dir, "code", "output", "target_spectrum.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

data = np.array(data, dtype=float)
sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]
freq_axis = sorted_data[:, 0]
yax = sorted_data[:, 1]# * 4000.0
plt.plot(freq_axis, yax, color='blue', label='Target Data')

with open(os.path.join(script_dir, "code", "output", "output_spectrum.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

data = np.array(data, dtype=float)
sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]
freq_axis = sorted_data[:, 0]
yax = sorted_data[:, 1]
plt.plot(freq_axis, yax, color='orange', label='Input dataset')
plt.title("Data Set Audio Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)

plt.xscale('log', base=10)
plt.xlim(20, 8000)
plt.legend()
plt.savefig(os.path.join(script_dir, "code", "output", "target_vs_output.png"))
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
#plt.ylim(0, 100000)

plt.savefig(os.path.join(script_dir, "code", "output", "target_spectrum.png"))
plt.clf()











with open(os.path.join(script_dir, "code", "output", "noise_red_vs_SNR_fac.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
xdata = data[:, 0]
ydata = data[:, 1]

bin_width = 0.1

y_median = []
y_std_minus = []
y_std_plus = []
binxax = []

for i in np.arange(0.0, (1.0-bin_width), 0.025):
    bin_mid = i + bin_width / 2.0
    binxax.append(bin_mid)
    mask = (xdata > i) & (xdata < (i+bin_width))
    xdata_new = xdata[mask]
    ydata_new = ydata[mask]
    y_median.append(np.median(ydata_new))
    y_std_minus.append(np.percentile(ydata_new, 15.8655253931457))
    y_std_plus.append(np.percentile(ydata_new, 84.1344746068543))

spl = interpolate.CubicSpline(binxax, y_median)
xnew = np.linspace(0.05, 0.95, num=1001)
plt.plot(xnew, spl(xnew))
plt.savefig(os.path.join(script_dir, "code", "output", "spline_median.png"))
plt.clf()

spl = interpolate.CubicSpline(binxax, y_std_plus)
xnew = np.linspace(0.05, 0.95, num=1001)
plt.plot(xnew, spl(xnew))
plt.savefig(os.path.join(script_dir, "code", "output", "spline_plus.png"))
plt.clf()

spl = interpolate.CubicSpline(binxax, y_std_minus)
xnew = np.linspace(0.05, 0.95, num=1001)
plt.plot(xnew, spl(xnew))
plt.savefig(os.path.join(script_dir, "code", "output", "spline_minus.png"))
plt.clf()






with open(os.path.join(script_dir, "code", "output", "noise_white_vs_SNR_dB.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
xdata = data[:, 0]
ydata = data[:, 1]
ydata = np.power(10, ydata/10.0)
ydata = ydata / (ydata + 1.0)
#file = open(os.path.join(script_dir, "code", "output", "debugg.txt"), 'w')
#for xyz in ydata:
    #file.write(str(xyz))
    #file.write("\n")
bin_width = 0.1
y_median_white = []
binxaxw = []
for i in np.arange(0.0, (1.0-bin_width), 0.025):
    bin_mid = i + bin_width / 2.0
    binxaxw.append(bin_mid)
    mask = (xdata > i) & (xdata < (i+bin_width))
    xdata_new = xdata[mask]
    ydata_new = ydata[mask]
    print(np.median(ydata_new))
    y_median_white.append(np.median(ydata_new))

with open(os.path.join(script_dir, "code", "output", "noise_pink_vs_SNR_dB.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
xdata = data[:, 0]
ydata = data[:, 1]
ydata = np.power(10, ydata/10.0)
ydata = ydata / (ydata + 1.0)
bin_width = 0.1
y_median_pink = []
binxaxp = []
for i in np.arange(0.0, (1.0-bin_width), 0.025):
    bin_mid = i + bin_width / 2.0
    binxaxp.append(bin_mid)
    mask = (xdata > i) & (xdata < (i+bin_width))
    xdata_new = xdata[mask]
    ydata_new = ydata[mask]
    y_median_pink.append(np.median(ydata_new))

with open(os.path.join(script_dir, "code", "output", "noise_shot_vs_SNR_dB.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
xdata = data[:, 0]
ydata = data[:, 1]
ydata = np.power(10, ydata/10.0)
ydata = ydata / (ydata + 1.0)
bin_width = 0.1
y_median_shot = []
binxaxs = []
for i in np.arange(0.0, (1.0-bin_width), 0.025):
    bin_mid = i + bin_width / 2.0
    binxaxs.append(bin_mid)
    mask = (xdata > i) & (xdata < (i+bin_width))
    xdata_new = xdata[mask]
    ydata_new = ydata[mask]
    y_median_shot.append(np.median(ydata_new))










with open(os.path.join(script_dir, "code", "output", "noise_red_vs_SNR_fac.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
xdata = data[:, 0]
ydata = data[:, 1]
plt.scatter(xdata, ydata, color='orange', label='Noise reduction', s=0.3)
"""with open(os.path.join(script_dir, "code", "output", "noise_red_vs_avg_SNR_fac.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]
xax = sorted_data[:, 0]
yax = sorted_data[:, 1]

spl = interpolate.CubicSpline(xax, yax)
xnew = np.linspace(0.05, 0.95, num=1001)
plt.plot(xnew, spl(xnew))"""
spl = interpolate.CubicSpline(binxax, y_median)
xnew = np.linspace(0.05, 0.95, num=1001)
plt.plot(xnew, spl(xnew), color='black', label='Median')
spl = interpolate.CubicSpline(binxax, y_std_plus)
xnew = np.linspace(0.05, 0.95, num=1001)
plt.plot(xnew, spl(xnew), color='black', label='+1 Standard Deviation')
spl = interpolate.CubicSpline(binxax, y_std_minus)
xnew = np.linspace(0.05, 0.95, num=1001)
plt.plot(xnew, spl(xnew), color='black', label='-1 Standard Deviation')
plt.legend()
plt.savefig(os.path.join(script_dir, "code", "output", "noise_red_vs_SNR_fac.png"))
plt.clf()










with open(os.path.join(script_dir, "code", "output", "noise_shot_vs_SNR_dB.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
shot_x = data[:, 0]
shot_x = np.power(10, (shot_x/10.0)) / ( np.power(10, (shot_x/10.0)) + 1.0 )
shot_y = data[:, 1]
indices = np.random.permutation(len(shot_x))
shot_x = shot_x[indices]
shot_y = shot_y[indices]

with open(os.path.join(script_dir, "code", "output", "noise_white_vs_SNR_dB.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
white_x = data[:, 0]
white_x = np.power(10, (white_x/10.0)) / ( np.power(10, (white_x/10.0)) + 1.0 )
white_y = data[:, 1]
indices = np.random.permutation(len(white_x))
white_x = white_x[indices]
white_y = white_y[indices]

with open(os.path.join(script_dir, "code", "output", "noise_pink_vs_SNR_dB.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
pink_x = data[:, 0]
pink_x = np.power(10, (pink_x/10.0)) / ( np.power(10, (pink_x/10.0)) + 1.0 )
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
        plt.scatter(val_pink_x, val_pink_y, color='blue', label='Pink Noise' if i == 0 else "", s=0.15, alpha=0.25)
    if i < len(white_x):
        plt.scatter(val_white_x, val_white_y, color='green', label='White Noise' if i == 0 else "", s=0.15, alpha=0.25)
    if i < len(shot_x):
        plt.scatter(val_shot_x, val_shot_y, color='red', label='Shot Noise' if i == 0 else "", s=0.15, alpha=0.25)

#plt.scatter(shot_x, shot_y, color='blue', label='Shot Noise', s=0.3, alpha=0.5)
#plt.scatter(pink_x, pink_y, color='red', label='Pink Noise', s=0.3, alpha=0.5)
#plt.scatter(white_x, white_y, color='green', label='White Noise', s=0.3, alpha=0.5)
plt.xlabel('SNR fac')
plt.ylabel('Noise reduction in dB')


spl = interpolate.CubicSpline(binxaxw, y_median_white)
xnew = np.linspace(0.05, 0.95, num=1001)
plt.plot(xnew, spl(xnew), color='green', label='Median White')
spl = interpolate.CubicSpline(binxaxp, y_median_pink)
xnew = np.linspace(0.05, 0.95, num=1001)
plt.plot(xnew, spl(xnew), color='blue', label='Median Pink')
spl = interpolate.CubicSpline(binxaxs, y_median_shot)
xnew = np.linspace(0.05, 0.95, num=1001)
plt.plot(xnew, spl(xnew), color='red', label='Median Shot')



#plt.title('Scatter Plot')
plt.legend()
plt.savefig(os.path.join(script_dir, "code", "output", "noise_reduction_by_type_fac.png"), dpi=600)
plt.clf()










# DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE
# DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE
# DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE
# DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE
# DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE
# DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE
# DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE - DB PART HERE

with open(os.path.join(script_dir, "code", "output", "noise_red_vs_SNR_dB.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
xdata = data[:, 0]
ydata = data[:, 1]

bin_width = 3.0

y_median = []
y_std_minus = []
y_std_plus = []
binxax = []

for i in np.arange((0.0 - SNR_RANGE), (SNR_RANGE-bin_width), 0.5):
    bin_mid = i + bin_width / 2.0
    binxax.append(bin_mid)
    mask = (xdata > i) & (xdata < (i+bin_width))
    xdata_new = xdata[mask]
    ydata_new = ydata[mask]
    y_median.append(np.median(ydata_new))
    y_std_minus.append(np.percentile(ydata_new, 15.8655253931457))
    y_std_plus.append(np.percentile(ydata_new, 84.1344746068543))











with open(os.path.join(script_dir, "code", "output", "noise_white_vs_SNR_dB.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
xdata = data[:, 0]
ydata = data[:, 1]
bin_width = 3.0
y_median_white = []
binxaxw = []
for i in np.arange((0.0 - SNR_RANGE), (SNR_RANGE-bin_width), 0.5):
    bin_mid = i + bin_width / 2.0
    binxaxw.append(bin_mid)
    mask = (xdata > i) & (xdata < (i+bin_width))
    xdata_new = xdata[mask]
    ydata_new = ydata[mask]
    y_median_white.append(np.median(ydata_new))

with open(os.path.join(script_dir, "code", "output", "noise_pink_vs_SNR_dB.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
xdata = data[:, 0]
ydata = data[:, 1]
bin_width = 3.0
y_median_pink = []
binxaxp = []
for i in np.arange((0.0 - SNR_RANGE), (SNR_RANGE-bin_width), 0.5):
    bin_mid = i + bin_width / 2.0
    binxaxp.append(bin_mid)
    mask = (xdata > i) & (xdata < (i+bin_width))
    xdata_new = xdata[mask]
    ydata_new = ydata[mask]
    y_median_pink.append(np.median(ydata_new))

with open(os.path.join(script_dir, "code", "output", "noise_shot_vs_SNR_dB.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
xdata = data[:, 0]
ydata = data[:, 1]
bin_width = 3.0
y_median_shot = []
binxaxs = []
for i in np.arange((0.0 - SNR_RANGE), (SNR_RANGE-bin_width), 0.5):
    bin_mid = i + bin_width / 2.0
    binxaxs.append(bin_mid)
    mask = (xdata > i) & (xdata < (i+bin_width))
    xdata_new = xdata[mask]
    ydata_new = ydata[mask]
    y_median_shot.append(np.median(ydata_new))








with open(os.path.join(script_dir, "code", "output", "noise_red_vs_SNR_dB.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
xdata = data[:, 0]
ydata = data[:, 1]
plt.scatter(xdata, ydata, color='orange', label='Noise reduction', s=0.3)
"""with open(os.path.join(script_dir, "code", "output", "noise_red_vs_avg_SNR_fac.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
sort_indices = np.argsort(data[:, 0])
sorted_data = data[sort_indices]
xax = sorted_data[:, 0]
yax = sorted_data[:, 1]

spl = interpolate.CubicSpline(xax, yax)
xnew = np.linspace(0.05, 0.95, num=1001)
plt.plot(xnew, spl(xnew))"""
spl = interpolate.CubicSpline(binxax, y_median)
xnew = np.linspace(-13.5, 13.5, num=1001)
plt.plot(xnew, spl(xnew), color='black', label='Median')
spl = interpolate.CubicSpline(binxax, y_std_plus)
xnew = np.linspace(-13.5, 13.5, num=1001)
plt.plot(xnew, spl(xnew), color='black', label='+1 Standard Deviation')
spl = interpolate.CubicSpline(binxax, y_std_minus)
xnew = np.linspace(-13.5, 13.5, num=1001)
plt.plot(xnew, spl(xnew), color='black', label='-1 Standard Deviation')
plt.xlim(-15.0, 15.0)
plt.legend()
plt.savefig(os.path.join(script_dir, "code", "output", "noise_red_vs_SNR_dB.png"))
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
        plt.scatter(val_pink_x, val_pink_y, color='blue', label='Pink Noise' if i == 0 else "", s=0.15, alpha=0.25)
    if i < len(white_x):
        plt.scatter(val_white_x, val_white_y, color='green', label='White Noise' if i == 0 else "", s=0.15, alpha=0.25)
    if i < len(shot_x):
        plt.scatter(val_shot_x, val_shot_y, color='red', label='Shot Noise' if i == 0 else "", s=0.15, alpha=0.25)

#plt.scatter(shot_x, shot_y, color='blue', label='Shot Noise', s=0.3, alpha=0.5)
#plt.scatter(pink_x, pink_y, color='red', label='Pink Noise', s=0.3, alpha=0.5)
#plt.scatter(white_x, white_y, color='green', label='White Noise', s=0.3, alpha=0.5)
plt.xlabel('SNR in dB')
plt.ylabel('Noise reduction in dB')

"""binxax = np.array(binxax)
binxax = 10.0 * np.log10(binxax / (1.0 - binxax))
rangefrom = 10.0 * np.log10(0.05 / (1.0 - 0.05))
rangeto = 10.0 * np.log10(0.95 / (1.0 - 0.95))"""

"""with open(os.path.join(script_dir, "code", "output", "noise_red_vs_SNR_dB.csv"), 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
data = np.array(data, dtype=float)
xdata = data[:, 0]
ydata = data[:, 1]

bins = np.linspace(-10, 10, 11)
bin_indices = np.digitize(xdata, bins) - 1
binned_data = [[] for _ in range(10)]
for i, bin_idx in enumerate(bin_indices):
    if (bin_idx > -1 and bin_idx < 10):
        binned_data[bin_idx].append((xdata[i], ydata[i]))
binned_data = [np.array(bin_data) for bin_data in binned_data]
y_median = []
y_std_minus = []
y_std_plus = []
binxax = []
for i in range(10):
    binxax.append(-9.0 + 2.0*i)
    bin_data = binned_data[i]
    y_median.append(np.median(bin_data[:, 1]))
    y_std_minus.append(np.percentile(bin_data[:, 1], 15.8655253931457))
    y_std_plus.append(np.percentile(bin_data[:, 1], 84.1344746068543))"""


spl = interpolate.CubicSpline(binxaxp, y_median_white)
xnew = np.linspace(-13.5, 13.5, num=1001)
plt.plot(xnew, spl(xnew), color='green', label='Median White')
spl = interpolate.CubicSpline(binxaxw, y_median_pink)
xnew = np.linspace(-13.5, 13.5, num=1001)
plt.plot(xnew, spl(xnew), color='blue', label='Median Pink')
spl = interpolate.CubicSpline(binxaxs, y_median_shot)
xnew = np.linspace(-13.5, 13.5, num=1001)
plt.plot(xnew, spl(xnew), color='red', label='Median Shot')

plt.xlim(-15.0, 15.0)

#plt.title('Scatter Plot')
plt.legend()
plt.savefig(os.path.join(script_dir, "code", "output", "noise_reduction_by_type_dB.png"), dpi=600)
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
x_data_3 = test_l1_data[:, 0]
y_data_3 = test_l1_data[:, 1]

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Steps')
ax1.set_ylabel('Test Loss', color=color)
#ax1.set_ylim(bottom=np.min(y_data_1), top=np.max(y_data_1))
ax1.plot(x_data_1, y_data_1, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Training Loss', color=color)
#ax2.set_ylim(bottom=np.min(y_data_2), top=np.max(y_data_2))
ax2.plot(x_data_2, y_data_2, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
plt.savefig(os.path.join(script_dir, "code", "output", "training_vs_test_loss.png"), dpi=600)
plt.clf()


fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Steps')
ax1.set_ylabel('Test Loss', color=color)
#ax1.set_ylim(bottom=np.min(y_data_1), top=np.max(y_data_1))
ax1.plot(x_data_1, y_data_1, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Test Loss L1', color=color)
#ax2.set_ylim(bottom=np.min(y_data_3), top=np.max(y_data_3))
ax2.plot(x_data_3, y_data_3, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
plt.savefig(os.path.join(script_dir, "code", "output", "test_l1_vs_test_loss.png"), dpi=600)
plt.clf()


fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Steps')
ax1.set_ylabel('Training Loss', color=color)
#ax1.set_ylim(bottom=np.min(y_data_2), top=np.max(y_data_2))
ax1.plot(x_data_2, y_data_2, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Test Loss L1', color=color)
#ax2.set_ylim(bottom=np.min(y_data_3), top=np.max(y_data_3))
ax2.plot(x_data_3, y_data_3, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
plt.savefig(os.path.join(script_dir, "code", "output", "training_vs_test_l1_loss.png"), dpi=600)
plt.clf()
















zip_file_path = os.path.join(script_dir, "code", "output")
folder_path = os.path.join(script_dir, "code", "output")
shutil.make_archive(zip_file_path, 'zip', folder_path)
print("Zipped all outputs")
print("Done!")