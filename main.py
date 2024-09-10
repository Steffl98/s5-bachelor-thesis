import struct
import math
import random
import os
import s5
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Tuple, Optional, Literal
Initialization = Literal['dense_columns', 'dense', 'factorized']

ITERATIONS = 101#int(37000*2 + 1)
STATE_DIM = 6
DIM = 24
LR = 0.01
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



class AudioDataSet(Dataset):
    def __init__(self, dir, transform=None, target_transform=None):
        self.dir = dir
        self.files = list_files(dir)
        #print(len(self.files))
        self.transform = transform
        self.target_transform = target_transform
        self.pink_noise = read_wav(os.path.join(script_dir, "..", "audio", "noise", "noise_pink_flicker_16k.wav"))
        self.shot_noise = read_wav(os.path.join(script_dir, "..", "audio", "noise", "Noise_Shot_16k.wav"))
        self.trans_noise = read_wav(os.path.join(script_dir, "..", "audio", "noise", "Noise_transistor_at_16k.wav"))
        self.white_noise = read_wav(os.path.join(script_dir, "..", "audio", "noise", "Noise_white_16k.wav"))
        self.SNR_fac = random.uniform(0.75, 1)#0.8#0.65
    def __len__(self):
        return ITERATIONS#100001#40000#8000#len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx % 852]
        offs = random.randint(0, 16000)
        shift = random.uniform(0, 2) * np.pi
        fshift = pow(1.2, random.uniform(-1, 1))
        freq = 100 * pow(2, random.uniform(0, 6.32))#100 Hz - 8 KHz (nyquist)
        amp = random.uniform(0, 1) / 3.0

        shift2 = random.uniform(0, 2) * np.pi
        fshift2 = pow(1.2, random.uniform(-1, 1))
        freq2 = 100 * pow(2, random.uniform(0, 6.32))  # 100 Hz - 8 KHz (nyquist)
        amp2 = random.uniform(0, 1) / 3.0

        shift3 = random.uniform(0, 2) * np.pi
        fshift3 = pow(1.2, random.uniform(-1, 1))
        freq3 = 100 * pow(2, random.uniform(0, 6.32))  # 100 Hz - 8 KHz (nyquist)
        amp3 = random.uniform(0, 1) / 3.0

        label_data = resample(read_wav(path), fshift*44.1/16.0, offs)
        #label_data = resample(label_data, 1.0)
        noise_choice = random.randint(1, 2)
        if (noise_choice == 1):
            audio_data = add_noise(label_data, self.pink_noise, self.SNR_fac)
        #if (noise_choice == 2):
            #audio_data = add_noise(label_data, self.shot_noise, self.SNR_fac)
        #if (noise_choice == 3):
#            audio_data = add_noise(label_data, self.trans_noise, self.SNR_fac)
        if (noise_choice == 2):
            audio_data = add_noise(label_data, self.white_noise, self.SNR_fac)
        label_data = amplify(label_data, self.SNR_fac)
        return (torch.tensor(audio_data)).unsqueeze(1), (torch.tensor(label_data)).unsqueeze(1)


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
        #self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        #self.rnn = nn.LSTM(input_size=1, hidden_size=15, num_layers=3, proj_size=1, dropout=0.3, batch_first=True)
        dim = DIM#5  # 100
        self.l1 = nn.Linear(1, dim)
        self.tanh = nn.Tanh()
        self.l2 = nn.Linear(dim, 1)

        state_dim = STATE_DIM#72# 160#64
        bidir = False
        self.s5 = s5.S5(dim, state_dim)#s5.S5Block(dim, state_dim, bidir)
        self.s5b = s5.S5(dim, state_dim)
        self.s5c = s5.S5(dim, state_dim)
        self.LN = torch.nn.LayerNorm((SAMPLE_LEN, dim))
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)


    def forward(self, x):
        h0 = (torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        out = self.l1(x.float())
        res = out.clone()
        out = self.s5(out)
        out = self.relu(out) + res

        res = out.clone()
        out = self.LN(out)
        out = self.s5b(out)
        out = self.relu(out) + res
        out = self.s5c(out)
        out = self.l2(out)
        return out



def train_model(tr_data, tr_model):
    train_dataloader = DataLoader(tr_data, batch_size=1, shuffle=True)

    test_dataloader = DataLoader(tr_data, batch_size=64, shuffle=True)

    loss_func = nn.MSELoss()


    optimizer = torch.optim.Adam(tr_model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    tr_model.train()
    epochs = 1
    for ep in range(epochs):
        it = 0
        print("\nEPOCH: ", ep)
        optimizer.zero_grad()
        loss_counter = 0.0
        for input, target in train_dataloader:
            it = it + 1

            output = tr_model(input)

            sq1 = torch.square(torch.sub(output, torch.mean(output)))
            sq2 = torch.square(torch.sub(target, torch.mean(target)))
            loss = 2*loss_func(output, target)# + loss_func(torch.mean(output), torch.mean(target)) + loss_func(torch.mean(sq1), torch.mean(sq2))
            loss_counter = loss_counter + loss.item()

            loss.backward()


            if (it % 100 == 0):
                optimizer.step()
                optimizer.zero_grad()
            if (it % 100 == 0):
                batches = ITERATIONS / 100
                print("\n        Iteration: ", it)
                print("            Loss: ", loss_counter/100.0)
                loss_counter = 0.0




script_dir = os.path.dirname(__file__)#os.path.dirname(os.path.abspath(__file__))
script_dir = "C:\\Users\\stefa\\OneDrive\\Desktop\\Uni\\Bachelorarbeit\\audio"


training_data = AudioDataSet(os.path.join(script_dir, "..", "audio", "voice_clips_wav"))
t1, t2 = training_data.__getitem__(69)

t_list = (torch.flatten(t1)).tolist()


model = SequenceToSequenceRNN(input_size=1, hidden_size=1)

train_model(training_data, model)

it = 0
test_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
for input, target in test_dataloader:
    it = it + 1
    output = model(input)
    t_list = (torch.flatten(target)).tolist()
    if (it > 1000):
        quit()
    with open(f"output\\{it}_tar.rawww", 'wb') as f:
        for i in range(len(t_list)):
            packed_data = struct.pack('<h', int(bound_f(t_list[i], -1.0, 1.0)*32767.5-0.5))
            f.write(packed_data)
    t_list = (torch.flatten(output)).tolist()
    with open(f"output\\{it}_out.rawww", 'wb') as f:
        for i in range(len(t_list)):
            packed_data = struct.pack('<h', int(bound_f(t_list[i], -1.0, 1.0)*32767.5-0.5))
            f.write(packed_data)
    t_list = (torch.flatten(input)).tolist()
    with open(f"output\\{it}_in.rawww", 'wb') as f:
        for i in range(len(t_list)):
            packed_data = struct.pack('<h', int(bound_f(t_list[i], -1.0, 1.0) * 32767.5 - 0.5))
            f.write(packed_data)