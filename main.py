#import faulthandler
#faulthandler.enable()
import struct
import math
import random
import os
import sys
import time
import s5
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Tuple, Optional, Literal
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import pandas as pd
import shutil
Initialization = Literal['dense_columns', 'dense', 'factorized']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

script_dir = os.path.join("C:\\", "Users", "stefa", "OneDrive", "Desktop", "Uni", "Bachelorarbeit")
try:
    argument1 = sys.argv[1]
    print("Using working directory: ", argument1)
    script_dir = argument1
except IndexError:
    print("Usage: python main.py <audio_dir_path>")
    print("Attempting to use default path...")
    #sys.exit(1)

ITERATIONS = 32*802*12#320128#38400#int(37000*2 + 1)
BATCH_SIZE = 32
NUM_WORKERS = 8
NUM_EPOCHS = 100
STATE_DIM = 8
DIM = 12
LR = 0.0025
SAMPLE_LEN = 32000

def bound_f(x, lower_bound=3.7, upper_bound=7.9):
    return max(lower_bound, min(x, upper_bound))

def data_to_type(data, data_type, byte_order='little'):
  """  Args:
    data_type: The data type of the integers (either 'h' for signed short or 'H' for unsigned short).
    byte_order: The byte order of the data (either 'little' or 'big').
  """
  if (byte_order == 'little'):
      byte_order = '<'
  else:
      byte_order = '>'
  return struct.unpack(byte_order + data_type, data)[0]


def read_wav(filename): # max len after resampling = 982988
    values = []
    with open(filename, 'rb') as f:
        data = f.read(44)
        while True:
            data = f.read(2)  # Read 2 bytes for 16-bit integers
            if not data:
                break
            value = (data_to_type(data, 'h')+0.5)/32767.5 # h = short, H = unsigned short; https://docs.python.org/3/library/struct.html
            values.append(value)
    return values

def resample(data, ratio, offset=0):
    #offset = 0
    xyz = []
    old_num = len(data)
    new_num = int(old_num / ratio)
    for i in range(new_num):
        indecks = int(i * ratio) + offset
        if (i >= SAMPLE_LEN): # C U L L
            break
        if (indecks < old_num):
            xyz.append(data[indecks])
    while (len(xyz) < SAMPLE_LEN): # P A D
        xyz.append(int(0))
    return xyz


def resample2(data, ratio, leng):
    #offset = 0
    xyz = []
    old_num = len(data)
    new_num = int(old_num / ratio)
    app_cnt = 0
    for i in range(new_num):
        indecks = int(i * ratio)
        if (i >= leng): # C U L L
            break
        if (indecks < old_num):
            xyz.append(data[indecks])
            app_cnt = app_cnt + 1
    while (len(xyz) < leng): # P A D
        xyz.append(int(0))
    return xyz, app_cnt


def add_noise(data, noise, fac):
    samples = len(data)
    noize = []
    for i in range(samples):
        noize.append((fac*data[i] + (1.0 - fac)*noise[i % samples]))
    return noize

def amplify(data, fac):
    samples = len(data)
    ret = []
    for i in range(samples):
        ret.append((fac*data[i]))
    return ret

def list_files(directory):
    files = []
    for entry in os.listdir(directory):
        path = os.path.join(directory, entry)
        if os.path.isfile(path):
            files.append(path)
    return files


def get_files_lists(dir, no_val, no_test):
    files = list_files(dir)
    random.shuffle(files)
    val_files = files[:no_val]
    files = files[no_val:]
    test_files = files[:no_test]
    files = files[no_test:]
    return files, val_files, test_files

class AudioDataSet(Dataset):
    def __init__(self, files, transform=None, target_transform=None):
        #self.dir = dir
        self.files = files#list_files(dir)
        #self.mode = mode
        self.wavs = []
        #self.wavs_test = []
        #self.wavs_val = []
        cntrr = 0
        for item in files:
            cntrr = cntrr + 1
            if (cntrr % 32 == 0):
                percstr = str( (cntrr * 100.0) / len(self.files) )
                if (len(percstr) > 4):
                    percstr = percstr[0:4]
                print("Loading wavs: ", percstr, "%")
            (self.wavs).append(read_wav(item))
        print("Done loading wavs.")
        #random.shuffle(self.wavs)
        #self.wavs_test = self.wavs[:200]
        #self.wavs = self.wavs[200:]
        self.transform = transform
        self.target_transform = target_transform
        self.pink_noise = read_wav(os.path.join(script_dir, "audio", "noise", "noise_pink_flicker_16k.wav"))
        self.shot_noise = read_wav(os.path.join(script_dir, "audio", "noise", "Noise_Shot_16k.wav"))
        self.trans_noise = read_wav(os.path.join(script_dir, "audio", "noise", "Noise_transistor_at_16k.wav"))
        self.white_noise = read_wav(os.path.join(script_dir, "audio", "noise", "Noise_white_16k.wav"))
        self.SNR_fac = []
        self.noise_choice = []
        self.fshift = []
        self.offs = []
        for _ in range(ITERATIONS):
            (self.SNR_fac).append(random.uniform(0.0, 1.0)) # formerly 0.75 - 1
            (self.noise_choice).append(random.randint(1, 3))
            (self.fshift).append(pow(1.2, random.uniform(-1, 1)))
            (self.offs).append(random.randint(0, 16000))
    def __len__(self):
        return ITERATIONS
    def __getitem__(self, idx):
        offs = self.offs[idx]
        fshift = self.fshift[idx]
        noice = self.noise_choice[idx]

        label_data = resample(self.wavs[idx % len(self.wavs)], fshift*44.1/16.0, offs)

        if (noice == 1):
            audio_data = add_noise(label_data, self.pink_noise, self.SNR_fac[idx])
        if (noice == 2):
            audio_data = add_noise(label_data, self.white_noise, self.SNR_fac[idx])
        if (noice == 3):
            audio_data = add_noise(label_data, self.shot_noise, self.SNR_fac[idx])
        label_data = amplify(label_data, self.SNR_fac[idx])
        return (torch.tensor(audio_data)).unsqueeze(1), (torch.tensor(label_data)).unsqueeze(1)
    def get_SNR_fac(self, x):
        return self.SNR_fac[x]
    def get_noise_choice(self, x):
        return self.noise_choice[x]
    #def set_mode(self, mode):
        #self.mode = mode


class SequenceToSequenceRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        #num_layers = 1
        super(SequenceToSequenceRNN, self).__init__()
        # N = batch size, L = sequence length, H_in = input_size, H_out = hidden_size, D whether bi
        """
        If your input data is of shape (seq_len, batch_size, features) then you don’t need batch_first=True and your
        LSTM will give output of shape (seq_len, batch_size, hidden_size).
        If your input data is of shape (batch_size, seq_len, features) then you need batch_first=True and your LSTM
        will give output of shape (batch_size, seq_len, hidden_size).
        """
        # Input: (L, H_in) unbatched / (L, N, H_in) / (N, L, H_in)
        # h_n: (D * num_layers, H_out) unbatched / (D * num_layers, N, H_out)
        # output: (L, D * H_out) unbatched / (L, N, D * H_out) / (N, L, D * H_out)
        # EXAMPLE: H_in = H_out = 1 because time series, L = samples, N = batch iteration; batch_first = True
        #     Input: (N, L, 1); Output: (N, L, 1); h_n: (num_layers, N, 1)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        dim = DIM
        self.l1 = nn.Linear(1, dim)
        self.tanh = nn.Tanh()
        self.l2 = nn.Linear(dim, 1)

        state_dim = STATE_DIM
        bidir = False
        self.s5 = s5.S5(dim, state_dim)
        self.s5b = s5.S5(dim, state_dim)
        self.s5c = s5.S5(dim, state_dim)
        self.LN = torch.nn.LayerNorm((SAMPLE_LEN, dim))
        self.BN = nn.BatchNorm1d(SAMPLE_LEN)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)


    def forward(self, x):
        h0 = (torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        out = self.l1(x.float())
        res = out.clone()
        #out = self.LN(out)
        out = self.BN(out)
        out = self.s5(out)
        out = self.relu(out) + res

        res = out.clone()

        out = self.s5b(out)
        out = self.relu(out) + res
        out = self.s5c(out)
        out = self.l2(out)
        return out



def train_model(tr_data, val_data, tr_model):
    #train_dataloader = DataLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    train_dataloader = DataLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    print("Initialized data loader.")
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=NUM_WORKERS,
                                  pin_memory=True)

    #test_dataloader = DataLoader(tr_data, batch_size=64, shuffle=True)

    loss_func = nn.MSELoss()
    print("Initialized loss func")


    optimizer = torch.optim.Adam(tr_model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4096, 8192, 12288, 16384], gamma=0.316227766)
    print("Initialized optimizer")

    tot_start_time = time.time()
    #epochs = NUM_EPOCHS#int(num_iterations)
    num_iterations = len(train_dataloader)
    print("Num iterations: ", num_iterations)
    loss_counter = 0.0
    cum_time = 0.0
    cum_err = 0.0
    iterations_list = []
    error_list = []
    test_db_list = []
    flog = open(os.path.join(script_dir, "code", "output", "output_log.txt"), "w")
    tr_model.train()
    for batch_idx, (data, target) in enumerate(train_dataloader):
        start_time = time.time()
        if (batch_idx % 400 == 0):
            print("Batch index: ", batch_idx)
        optimizer.zero_grad()
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output = tr_model(data)
        loss = loss_func(output, target)
        loss_counter = loss_counter + loss.item()
        loss.backward()
        optimizer.step()
        #scheduler.step()
        end_time = time.time()
        cum_time = cum_time + (end_time - start_time)
        cum_err = cum_err + math.log(loss.item())
        if (batch_idx % 400 == 0):
            print("Error (log): ", cum_err / 400.0, "  ; took ", cum_time, " seconds...")
            flog.write(f"{cum_err / 400.0}\t{cum_time}\n")
            iterations_list.append(batch_idx)
            error_list.append(cum_err / 400.0)
            cum_time = 0
            cum_err = 0
            tr_model.eval()
            val_loss = 0.0
            nsamples = 0
            zeros = [0] * SAMPLE_LEN
            zeros = ((torch.tensor(zeros)).unsqueeze(1)).unsqueeze(0)
            zeros = zeros.to(device, non_blocking=True)
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    nsamples = nsamples + 1
                    if (nsamples > 100):
                        break
                    outputs = tr_model(inputs)
                    noise_remaining = 10.0 * math.log10(loss_func((outputs - labels), zeros).item())
                    SNR_fac = val_data.get_SNR_fac(nsamples - 1)
                    fac_noise_red = 10.0 * math.log10(1.0 - SNR_fac)
                    noise_db = 0.0
                    noice = val_data.get_noise_choice(nsamples - 1)
                    if (noice == 1):
                        noise_db = -15.9789
                    if (noice == 2):
                        noise_db = -7.77903
                    if (noice == 3):
                        noise_db = -15.6357
                    val_loss = val_loss + (noise_remaining - noise_db - fac_noise_red)
            val_loss /= 100.0  # /= len(val_dataloader.dataset)
            print(f"Noise reduction in dB: {val_loss:.4f}")
            test_db_list.append(val_loss)
            tr_model.train()

    tot_end_time = time.time()
    print("In Total took ", (tot_end_time - tot_start_time), " seconds...")
    flog.write(str(tot_end_time - tot_start_time))
    flog.close()
    fig = go.Figure(data=go.Scatter(x=iterations_list, y=error_list, mode='markers'))
    fig.update_layout(title="Training performance", xaxis_title="Batch no.", yaxis_title="Ln of training loss")
    pio.write_image(fig, os.path.join(script_dir, "code", "output", "training_loss.png"), format="png")
    fig = go.Figure(data=go.Scatter(x=iterations_list, y=test_db_list, mode='markers'))
    fig.update_layout(title="Test performance", xaxis_title="Batch no.", yaxis_title="Noise reduction / dB")
    pio.write_image(fig, os.path.join(script_dir, "code", "output", "test_performance.png"), format="png")



def create_dataset_spectrogram():
    files = list_files(os.path.join(script_dir, "audio", "voice_clips_wav"))
    cntrr = 0
    fft_cum = np.array([0] * 50000)
    for item in files:
        cntrr = cntrr + 1
        if (cntrr % 32 == 0):
            percstr = str((cntrr * 100.0) / len(files))
            if (len(percstr) > 4):
                percstr = percstr[0:4]
            print("Loading wavs: ", percstr, "%")
        wav = read_wav(item)
        label_data, samplecnt = resample2(wav, 44.1 / 16.0, 50000)
        #t_list = (torch.flatten(label_data)).tolist()
        audio_data_np = np.array(label_data) * (16000.00 / float(samplecnt))#t_list)
        fft_result = fft(audio_data_np)
        fft_cum = fft_cum + np.abs(fft_result)

    audio_data_np = np.array(label_data)
    sampling_rate = 16000.0
    freq_axis = np.fft.fftfreq(len(audio_data_np), 1.0 / sampling_rate)
    plt.plot(freq_axis, np.abs(fft_cum))
    plt.title("Unadulterated Data Set Audio Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.xlim(0, 2000)
    plt.xscale('log', base=10)
    plt.xlim(20, 8000)
    plt.ylim(0, 20000)
    plt.savefig(os.path.join(script_dir, "code", "output", "data_set_spectrum.png"))
    plt.clf()

    with open(script_dir, "code", "output", "freq_axis.txt", "w") as f:
        for freq in freq_axis:
            f.write(str(freq) + "\n")

#script_dir = os.path.dirname(__file__)
#script_dir = "C:\\Users\\stefa\\OneDrive\\Desktop\\Uni\\Bachelorarbeit\\audio"

wav_lens = []
all_files = list_files(os.path.join(script_dir, "audio", "voice_clips_wav"))
for i in all_files:
    size_in_secs = (os.path.getsize(i)-44)/(2.0*44100.0)
    wav_lens.append(size_in_secs)

plt.hist(wav_lens, bins=120, color='blue')
plt.title("Histogram of Audio file lengths")
plt.xlabel("Seconds")
plt.ylabel("Frequency")
plt.savefig(os.path.join(script_dir, "code", "output", "histogram.png"))
plt.clf()

#create_dataset_spectrogram()
#quit()

files, val_files, test_files = get_files_lists(os.path.join(script_dir, "audio", "voice_clips_wav"), 100, 0)
training_data = AudioDataSet(files)
val_data = AudioDataSet(val_files)
test_data = AudioDataSet(test_files)
print("Finished preparing training data.")



power_list = []

zeros = [0] * SAMPLE_LEN
zeros = ((torch.tensor(zeros)).unsqueeze(1)).unsqueeze(0)
zeros = zeros.to(device, non_blocking=True)
iteration_dataloader = DataLoader(training_data, batch_size=1, shuffle=False)
loss_func = nn.MSELoss()
indx = 0
for input, target in iteration_dataloader:
    indx = indx + 1
    if (indx > len(files)):
        break
    target = target.to(device, non_blocking=True)
    power_db = 10.0 * math.log10(loss_func(target, zeros).item())
    power_list.append(power_db)


plt.hist(power_list, bins=88, color='blue')
plt.title("Histogram of Audio files avg. power")
plt.xlabel("dB")
plt.ylabel("Frequency")
plt.savefig(os.path.join(script_dir, "code", "output", "power_hist.png"))
plt.clf()



model = SequenceToSequenceRNN(input_size=1, hidden_size=1).to(device)
print("Finished preparing model.")

train_model(training_data, val_data, model)
torch.save(model.state_dict(), os.path.join(script_dir, "code", "output", "my_model.pth"))
#model.load_state_dict(torch.load(os.path.join(script_dir, "code", "output", "my_model.pth")))
print("Done training model.")

it = 0
#testing_data = copy.deepcopy(training_data)
#training_data.mode(2)
test_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
print("Initialized data loader.")

zeros = [0] * SAMPLE_LEN
zeros = ((torch.tensor(zeros)).unsqueeze(1)).unsqueeze(0)
zeros = zeros.to(device, non_blocking=True)
idx = 0
#fstat = open(os.path.join(script_dir, "code", "output", "statistics.txt"), "w")
loss_func = nn.MSELoss()
plot1x = []
plot1y = []
plot2x = []
plot2y = []
plot3x = []
plot3y = []
plot4x = []
plot4y = []
plot5x = []
plot5y = []
plot6x = []
plot6y = []
percentiles_val = [0.0]*10
percentiles_count = [0]*10

df = pd.DataFrame(columns=['SNR fac','SNR / dB','Noise remaining dB','Target dB','Output dB','Noise Type'])
cum_target_flag = 0
cum_output_flag = 0
cum_input_flag = 0
fft_target_cum = np.array([0] * SAMPLE_LEN)
fft_input_cum = np.array([0] * SAMPLE_LEN)
fft_output_cum = np.array([0] * SAMPLE_LEN)
for input, target in test_dataloader:
    input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
    it = it + 1
    output = model(input)
    if (cum_target_flag == 0):
        cum_target_flag = 1
        cum_target = target
    else:
        cum_target = cum_target + target
    if (cum_output_flag == 0):
        cum_output_flag = 1
        cum_output = output
    else:
        cum_output = cum_output + output
    if (cum_input_flag == 0):
        cum_input_flag = 1
        cum_input = input
    else:
        cum_input = cum_input + input

    t_list = (torch.flatten(target)).tolist()
    audio_data_np = np.array(t_list)
    fft_result = fft(audio_data_np)
    fft_target_cum = fft_target_cum + np.abs(fft_result)

    t_list = (torch.flatten(output)).tolist()
    audio_data_np = np.array(t_list)
    fft_result = fft(audio_data_np)
    fft_output_cum = fft_output_cum + np.abs(fft_result)

    t_list = (torch.flatten(input)).tolist()
    audio_data_np = np.array(t_list)
    fft_result = fft(audio_data_np)
    fft_input_cum = fft_input_cum + np.abs(fft_result)

    if (it > 2000):
        t_list = (torch.flatten(target)).tolist()
        audio_data_np = np.array(t_list)
        sampling_rate = 16000.0
        freq_axis = np.fft.fftfreq(len(audio_data_np), 1.0 / sampling_rate)
        plt.plot(freq_axis, np.abs(fft_target_cum))
        plt.title("Target Audio Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.xlim(0, 2000)
        plt.xscale('log', base=10)
        plt.xlim(20, 8000)
        plt.ylim(0, 48000)
        plt.savefig(os.path.join(script_dir, "code", "output", "target_spectrum.png"))
        plt.clf()

        t_list = (torch.flatten(input)).tolist()
        audio_data_np = np.array(t_list)
        sampling_rate = 16000.0
        freq_axis = np.fft.fftfreq(len(audio_data_np), 1.0 / sampling_rate)
        plt.plot(freq_axis, np.abs(fft_input_cum))
        plt.title("Input Audio Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.xlim(0, 2000)
        plt.xscale('log', base=10)
        plt.xlim(20, 8000)
        plt.ylim(0, 140000)
        plt.savefig(os.path.join(script_dir, "code", "output", "input_spectrum.png"))
        plt.clf()

        t_list = (torch.flatten(output)).tolist()
        audio_data_np = np.array(t_list)
        sampling_rate = 16000.0
        freq_axis = np.fft.fftfreq(len(audio_data_np), 1.0 / sampling_rate)
        plt.plot(freq_axis, np.abs(fft_output_cum))
        plt.title("Output Audio Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.xlim(0, 2000)
        plt.xscale('log', base=10)
        plt.xlim(20, 8000)
        plt.ylim(0, 48000)
        plt.savefig(os.path.join(script_dir, "code", "output", "output_spectrum.png"))
        plt.clf()

        t_list = (torch.flatten(output)).tolist()
        audio_data_np = np.array(t_list)
        sampling_rate = 16000.0
        freq_axis = np.fft.fftfreq(len(audio_data_np), 1.0 / sampling_rate)
        plt.plot(freq_axis, np.log10(fft_output_cum / fft_input_cum))
        plt.title("Transfer function Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude Log10")
        plt.grid(True)
        plt.xlim(0, 2000)
        plt.xscale('log', base=10)
        plt.xlim(20, 8000)
        plt.ylim(-3, 3)
        plt.savefig(os.path.join(script_dir, "code", "output", "transfer_function.png"))
        plt.clf()

        """t_list = (torch.flatten(cum_output)).tolist()
        audio_data_np = np.array(t_list)
        fft_result = fft(audio_data_np)
        fft_output = fft_result
        sampling_rate = 16000.0
        freq_axis = np.fft.fftfreq(len(audio_data_np), 1.0 / sampling_rate)
        plt.plot(freq_axis, np.abs(fft_result))
        plt.title("Output Audio Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.xlim(0, 2000)
        plt.xscale('log', base=10)
        plt.xlim(20, 8000)
        plt.ylim(0, 2000)
        plt.savefig(os.path.join(script_dir, "code", "output", "output_spectrum.png"))
        plt.clf()

        t_list = (torch.flatten(cum_input)).tolist()
        audio_data_np = np.array(t_list)
        fft_result = fft(audio_data_np)
        fft_input = fft_result
        sampling_rate = 16000.0
        freq_axis = np.fft.fftfreq(len(audio_data_np), 1.0 / sampling_rate)
        plt.plot(freq_axis, np.abs(fft_result))
        plt.title("Input Audio Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        #plt.xlim(0, 2000)
        plt.xscale('log', base=10)
        plt.xlim(20, 8000)
        plt.ylim(0, 2000)
        plt.savefig(os.path.join(script_dir, "code", "output", "input_spectrum.png"))
        plt.clf()

        t_list = (torch.flatten(cum_target - cum_output)).tolist()
        audio_data_np = np.array(t_list)
        fft_result = fft(audio_data_np)
        sampling_rate = 16000.0
        freq_axis = np.fft.fftfreq(len(audio_data_np), 1.0 / sampling_rate)
        plt.plot(freq_axis, np.abs(fft_result))
        plt.title("Target-Output Difference Audio Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        # plt.xlim(0, 2000)
        plt.xscale('log', base=10)
        plt.xlim(20, 8000)
        plt.ylim(0, 2000)
        plt.savefig(os.path.join(script_dir, "code", "output", "difference_spectrum.png"))
        plt.clf()

        fft_result = fft_input - fft_output
        plt.plot(freq_axis, fft_result)
        plt.title("Transfer Function")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.xscale('log', base=10)
        plt.xlim(20, 8000)
        plt.ylim(-2000, 2000)
        plt.savefig(os.path.join(script_dir, "code", "output", "transfer_function.png"))
        plt.clf()



        fft_result = fft_target - fft_output
        plt.plot(freq_axis, fft_result)
        plt.title("Difference Function")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.xscale('log', base=10)
        plt.xlim(20, 8000)
        plt.ylim(-2000, 2000)
        plt.savefig(os.path.join(script_dir, "code", "output", "difference_function.png"))
        plt.clf()"""





        #fstat.close()
        fig = go.Figure(data=go.Scatter(x=plot1x, y=plot1y, mode='markers'))
        fig.update_layout(title="Scatter plot", xaxis_title="SNR fac", yaxis_title="noise reduction in dB")
        pio.write_image(fig, os.path.join(script_dir, "code", "output", "plot.png"), format="png")

        fig = go.Figure(data=go.Scatter(x=plot2x, y=plot2y, mode='markers'))
        fig.update_layout(title="Scatter plot", xaxis_title="SNR fac in dB", yaxis_title="noise reduction in dB")
        pio.write_image(fig, os.path.join(script_dir, "code", "output", "plot2.png"), format="png")

        for i in range(10):
            plot3x.append((i/10.0) + 0.05)
            plot3y.append(percentiles_val[i] / percentiles_count[i])

        fig = go.Figure(data=go.Scatter(x=plot3x, y=plot3y, mode='markers'))
        fig.update_layout(title="Scatter plot", xaxis_title="avg. SNR fac", yaxis_title="avg. noise reduction in dB")
        pio.write_image(fig, os.path.join(script_dir, "code", "output", "plot3.png"), format="png")

        fig = go.Figure(data=go.Scatter(x=plot4x, y=plot4y, mode='markers'))
        fig.update_layout(title="Noise Type 1: Pink", xaxis_title="SNR fac in dB", yaxis_title="noise reduction in dB")
        pio.write_image(fig, os.path.join(script_dir, "code", "output", "plot4.png"), format="png")

        fig = go.Figure(data=go.Scatter(x=plot5x, y=plot5y, mode='markers'))
        fig.update_layout(title="Noise Type 2: White", xaxis_title="SNR fac in dB", yaxis_title="noise reduction in dB")
        pio.write_image(fig, os.path.join(script_dir, "code", "output", "plot5.png"), format="png")

        fig = go.Figure(data=go.Scatter(x=plot6x, y=plot6y, mode='markers'))
        fig.update_layout(title="Noise Type 3: Shot", xaxis_title="SNR fac in dB", yaxis_title="noise reduction in dB")
        pio.write_image(fig, os.path.join(script_dir, "code", "output", "plot6.png"), format="png")


        df.to_csv(os.path.join(script_dir, "code", "output", "validation_data.csv"), index=False)
        break
    noise_remaining = 10.0 * math.log10(loss_func((output - target), zeros).item())
    output_db = 10.0 * math.log10(loss_func((output), zeros).item())
    target_db = 10.0 * math.log10(loss_func((target), zeros).item())
    SNR_fac = training_data.get_SNR_fac(idx)
    SNR_db = 10.0 * math.log10(SNR_fac / (1.0 - SNR_fac))

    fac_noise_red = 10.0 * math.log10(1.0 - SNR_fac)
    noise_db = 0.0
    noice = training_data.get_noise_choice(idx)
    if (noice == 1):
        noise_db = -15.9789
        plot4x.append(SNR_db)
        plot4y.append(noise_remaining - noise_db - fac_noise_red)
    if (noice == 2):
        noise_db = -7.77903
        plot5x.append(SNR_db)
        plot5y.append(noise_remaining - noise_db - fac_noise_red)
    if (noice == 3):
        noise_db = -15.6357
        plot6x.append(SNR_db)
        plot6y.append(noise_remaining - noise_db - fac_noise_red)
    plot1x.append(SNR_fac)
    plot1y.append(noise_remaining - noise_db - fac_noise_red)
    plot2x.append(SNR_db)
    plot2y.append(noise_remaining - noise_db - fac_noise_red)

    prc = math.floor(SNR_fac * 9.999999)  # tmp is int and ranges from 0 to 9
    percentiles_count[prc] = percentiles_count[prc] + 1
    percentiles_val[prc] = percentiles_val[prc] + (noise_remaining - noise_db - fac_noise_red)

    pdrow=pd.DataFrame([[SNR_fac, SNR_db, noise_remaining,target_db,output_db,noice]],
                       columns=['SNR fac','SNR / dB','Noise remaining dB','Target dB','Output dB','Noise Type'])
    #df = df.append(pdrow, ignore_index=True)
    df = pd.concat([df, pdrow])
    #fstat.write(f"{SNR_fac}\t{noise_remaining}\t{target_db}\t{output_db}\n")
    idx = idx + 1
    if (it < 31):
        print("Saving file #", it)
        t_list = (torch.flatten(target)).tolist()
        with open(os.path.join(script_dir, "code", "output", f"{it}_tar.rawww"), 'wb') as f:
            for i in range(len(t_list)):
                packed_data = struct.pack('<h', int(bound_f(t_list[i], -1.0, 1.0)*32767.5-0.5))
                f.write(packed_data)
        t_list = (torch.flatten(output)).tolist()
        with open(os.path.join(script_dir, "code", "output", f"{it}_out.rawww"), 'wb') as f:
            for i in range(len(t_list)):
                packed_data = struct.pack('<h', int(bound_f(t_list[i], -1.0, 1.0)*32767.5-0.5))
                f.write(packed_data)
        t_list = (torch.flatten(input)).tolist()
        with open(os.path.join(script_dir, "code", "output", f"{it}_in.rawww"), 'wb') as f:
            for i in range(len(t_list)):
                packed_data = struct.pack('<h', int(bound_f(t_list[i], -1.0, 1.0) * 32767.5 - 0.5))
                f.write(packed_data)

zip_file_path = os.path.join(script_dir, "code", "output")
folder_path = os.path.join(script_dir, "code", "output")
shutil.make_archive(zip_file_path, 'zip', folder_path)
print("Zipped all outputs")
print("Done!")