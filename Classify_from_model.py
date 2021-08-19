import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import torchvision.transforms as transforms
from tqdm import tqdm
import random
import time
import argparse
import torch.utils.data
import torch.nn.utils.rnn
import matplotlib.pyplot as plt
from tensorboard_utils import CustomSummaryWriter
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from csv_utils_2 import CsvUtils2
from file_utils import FileUtils
from datetime import datetime
import os
from sklearn.model_selection import ParameterGrid
import torch_optimizer as optim

print('loading dataset...')
df = pd.read_pickle('datasets/MDB270721_cached.pkl').fillna(value=0, axis=0)
df2 = df[(df['Class_ID'] != 999) & (df['Class_ID'] != 99)].reset_index(drop=True)
print('loading done, proceeding')
sequence_len = 12

class2idx = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    11: 9,
    12: 10,
    13: 11,
    14: 12,
    15: 13,
    17: 14,
    18: 15,
    19: 16,
    20: 17
}

df2['Class_ID'].replace(class2idx, inplace=True)


class DatasetTabular(torch.utils.data.Dataset):
    def __init__(self, is_train, sequence_len):
        self.scaler = MinMaxScaler()
        self.is_train = is_train
        self.sequence_length = sequence_len
        self.data = df2
        self.data['Class_ID'] = self.data['Class_ID'].astype(int)
        self.data_x = self.data.iloc[:, 1:17]
        self.scaler.fit(self.data_x)
        self.data_y = self.data.iloc[:, 18:19]
        _ = ""

    def __len__(self):
        return int(len(self.data) / self.sequence_length)

    def __getitem__(self, idx):
        x1 = (self.data_x[(idx * sequence_len):((idx * sequence_len) + sequence_len)]).values
        x = np.array(x1, dtype="float32")
        x = self.scaler.transform(x)
        x = torch.from_numpy(x)
        y = int(self.data_y['Class_ID'][idx * sequence_len])
        y = torch.tensor([int(y)])

        # If use OHE
        # y = target_to_oh(y_idx)
        return x, y


data_loader_pred = torch.utils.data.DataLoader(
    dataset=DatasetTabular(is_train=False, sequence_len=sequence_len),
    batch_size=32,
    shuffle=True,
    drop_last=True,
)


class ModelBiLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.ff = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=input_size),
            torch.nn.Linear(in_features=input_size, out_features=hidden_size),
            torch.nn.BatchNorm1d(num_features=hidden_size)
        )
        self.rnn = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.5,
            bidirectional=True
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=hidden_size*2*2),
            torch.nn.Linear(in_features=hidden_size*2*2, out_features=output_size),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        lengths = torch.tensor([len(t) for t in x])
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
        x_internal = self.ff.forward(x_packed.data)
        x_repacked = torch.nn.utils.rnn.PackedSequence(
            data=x_internal,
            batch_sizes=x_packed.batch_sizes,
            sorted_indices=x_packed.sorted_indices,
            unsorted_indices=x_packed.unsorted_indices
        )

        h_packed, _ = self.rnn.forward(x_repacked)
        h_unpacked, len_unpacked = torch.nn.utils.rnn.pad_packed_sequence(h_packed, batch_first=True)
        h_stack = []

        for idx, h in enumerate(h_unpacked):
            h0_max, _ = torch.max(h[:len_unpacked[idx], :self.hidden_size], dim=0)
            h1_max, _ = torch.max(h[:len_unpacked[idx], self.hidden_size:], dim=0)

            h_stack.append(torch.stack([
                h[len_unpacked[idx]-1, :self.hidden_size],
                h[0, self.hidden_size:],
                h0_max,
                h1_max
            ]))

        h_stack = torch.stack(h_stack).view(x.size(0), -1)
        y_prime = self.mlp.forward(h_stack)
        return y_prime


model = ModelBiLSTM(input_size=16, hidden_size=64, num_layers=1,
                    output_size=18)
model.load_state_dict(torch.load('results/RNN_LSTM_CSV_2021_180821_full_dataset/run_1629295959.0764098-21-08-18--14-12-39/run_1629295959.0764098-21-08-18--14-12-39-model-22.pt'))
model.eval()

with torch.no_grad():
    for x, y in data_loader_pred:
        prediction = model(x)
        predicted_classes = np.argmax(prediction, axis=1)





