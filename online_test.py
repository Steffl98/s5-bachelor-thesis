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
from scipy.fftpack import fft
import shutil
Initialization = Literal['dense_columns', 'dense', 'factorized']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import csv

DIM = 8
STATE_DIM = 12
SAMPLE_LEN = 32000


class SequenceToSequenceS5(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SequenceToSequenceS5, self).__init__()

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
        self.LN = torch.nn.LayerNorm(dim)
        #self.BN = nn.BatchNorm1d(dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)
    def forward(self, x):
        h0 = (torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        out = self.l1(x.float())
        res = out.clone()
        #out = self.LN(out)
        out = self.LN(out)
        out = self.s5(out)
        out = self.relu(out) + res

        res = out.clone()

        out = self.s5b(out)
        out = self.relu(out) + res
        out = self.s5c(out)
        out = self.l2(out)
        return out


#x = torch.rand([2, 768, 32])  # (batch_size, sequence_length, input_dim)
#model = s5.S5(32, 128)           # Instantiate model with 32 input and 128 state dims
#model = SequenceToSequenceS5(69, 69)

#with torch.no_grad():
#    # Process sequences with state
#    res, state = model(x, return_state=True)
#    print(res.shape, state.shape)


# Define the RandomDataset class
class RandomDataset(Dataset):
    def __init__(self, num_samples, sequence_length, input_dim):
        self.num_samples = num_samples
        self.sequence_length = SAMPLE_LEN#sequence_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random input tensor
        x = torch.rand(self.sequence_length, 1)
        # Random target tensor (same shape as input for sequence-to-sequence tasks)
        y = torch.rand(self.sequence_length, 1)
        return x, y

def train_model():
    # Hyperparameters
    input_dim = 32
    hidden_size = 69
    num_layers = 1
    sequence_length = 768
    batch_size = 2
    num_steps = 10
    learning_rate = 0.001

    # Initialize the model
    model = SequenceToSequenceS5(input_dim, hidden_size, num_layers)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create a DataLoader with random data
    dataset = RandomDataset(num_samples=100, sequence_length=sequence_length, input_dim=input_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    for step in range(num_steps):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            # Forward pass
            output = model(x)

            # Compute loss
            loss = criterion(output, y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Step [{step+1}/{num_steps}], Loss: {loss.item():.4f}")

            # Break after one batch per step for demonstration purposes
            break

    # Save the model
    save_path = os.path.join("C:\\Users\\stefa\\OneDrive\\Desktop\\Uni\\Bachelorarbeit", "test_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Run the training script
if __name__ == "__main__":
    train_model()
    print("DANONE!")