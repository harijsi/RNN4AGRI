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


param_grid = {'-learning_rate': [1e-2, 1e-3], '-batch_size': [32, 64, 128],
              '-hidden_size': [32, 64, 128], '-num_layers': [1, 2]}

g_id = 0
num_of_classes = 6
input_size = 30

for g in tqdm(ParameterGrid(param_grid)):

    g_id += 1
    for k, v in g.items():
        if k == '-learning_rate':
            g_learning_rate = v
        if k == '-batch_size':
            g_batch_size = v
        if k == '-hidden_size':
            g_hidden_size = v
        if k == '-num_layers':
            g_num_layers = v

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('-id', default=g_id, type=int)
    parser.add_argument('-run_name', default=f'run_{time.time()}', type=str)
    parser.add_argument('-sequence_name', default=f'RNN_LSTM_CSV_S1_S2_best_metric_test', type=str)
    parser.add_argument('-learning_rate', default=g_learning_rate, type=float)
    parser.add_argument('-batch_size', default=g_batch_size, type=int)
    parser.add_argument('-epochs', default=1500, type=int)
    parser.add_argument('-is_cuda', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-sequence_len', default=25, type=int)
    parser.add_argument('-hidden_size', default=g_hidden_size, type=int)
    parser.add_argument('-num_layers', default=g_num_layers, type=int)
    args = parser.parse_args()

    path_sequence = f'./results/{args.sequence_name}'
    args.run_name += ('-' + datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))
    path_run = f'./results/{args.sequence_name}/{args.run_name}'
    path_artificats = f'./artifacts/{args.sequence_name}/{args.run_name}'

    FileUtils.createDir(path_run)
    FileUtils.createDir(path_artificats)
    FileUtils.writeJSON(f'{path_run}/args.json', args.__dict__)

    CsvUtils2.create_global(path_sequence)
    CsvUtils2.create_local(path_sequence, args.run_name)

    # Transform labels to one-hot
    def target_to_oh(target):
        num_class = num_of_classes
        one_hot = torch.eye(num_class)[target]
        return one_hot


    dataset = 'datasets/Processed_S1_S2_overfit_[1, 2, 3, 11, 12, 15]].csv'


    class DatasetTabular(torch.utils.data.Dataset):
        def __init__(self, is_train, sequence_len):
            self.scaler = MinMaxScaler()
            self.is_train = is_train
            self.sequence_length = sequence_len
            self.data = pd.read_csv(dataset).fillna(value=0, axis=0)
            self.data['Class_ID'] = self.data['Class_ID'].astype(int)
            self.data_x = self.data.iloc[:, 1:31]
            self.scaler.fit(self.data_x)
            self.data_y = self.data.iloc[:, 32:33]

        def __len__(self):
            return int(len(self.data) / self.sequence_length)

        def __getitem__(self, idx):
            x1 = (self.data_x[(idx * args.sequence_len):((idx * args.sequence_len) + args.sequence_len)]).values
            x = np.array(x1, dtype="float32")
            x = self.scaler.transform(x)
            y = int(self.data_y['Class_ID'][idx * args.sequence_len])

            # If use OHE
            # y = target_to_oh(y_idx)
            return x, y


    # Define parameters for train-test split
    torch.manual_seed(1)
    dataset_len = len(pd.read_csv(dataset)) / args.sequence_len
    train_size = int(0.8 * dataset_len)
    test_size = int(dataset_len - train_size)

    # Create train & test datasets through random split
    train_dataset, test_dataset = torch.utils.data.random_split(
        DatasetTabular(is_train=True, sequence_len=args.sequence_len),
        [train_size, test_size])

    data_loader_train = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )


    class LossCrossEntropy(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, y, y_prim):
            return -torch.mean(y * torch.log(y_prim + 1e-20))


    class ModelLSTM(torch.nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers):
            super().__init__()
            self.hidden_size = hidden_size
            self.linear = torch.nn.Linear(in_features=hidden_size, out_features=output_size)
            self.num_layers = num_layers

            self.rnn = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                num_layers=self.num_layers
            )

        def forward(self, input):
            # input - (B, S, F)
            rnn_out = self.rnn.forward(input)
            rnn_out = torch.mean(rnn_out[0], dim=1)
            rnn_out = self.linear.forward(rnn_out)
            output = torch.nn.functional.softmax(rnn_out, dim=1)

            return output

    model = ModelLSTM(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.num_layers,
                      output_size=num_of_classes)
    loss_func = LossCrossEntropy()
    optimizer = optim.RAdam(model.parameters(), lr=args.learning_rate)

    # if cuda move model and loss function to gpu
    if args.is_cuda:
        model = model.cuda()
        loss_func = loss_func.cuda()

    metrics = {}
    for stage in ['train', 'test']:
        for metric in [
            'loss', 'acc',
        ]:
            metrics[f'{stage}_{metric}'] = 0
        if stage == 'test':
            metrics[f'best_{stage}_{metric}'] = 0

    # Variables to keep track of
    best_test_acc = 0.0
    best_test_loss = ''
    elapsed = 0
    patience = 50

    for epoch in range(0, args.epochs):

        for data_loader in [data_loader_train, data_loader_test]:
            metrics_epoch = {key: [] for key in metrics.keys()}

            stage = 'train'
            if data_loader == data_loader_test:
                stage = 'test'

            for x, y in data_loader:
                # hidden = model.init_hidden(x.size(0))
                # if cuda move data to cuda device
                if args.is_cuda:
                    x = x.cuda()
                    y = y.cuda()

                y_prim = model.forward(x)

                idxes = torch.arange(0, args.batch_size).to('cuda')
                loss = -torch.mean(torch.log(y_prim[idxes, y[idxes]] + 1e-20))
                # loss = -torch.mean(y * torch.log(y_prim + 1e-20))

                if data_loader == data_loader_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # move all data back to cpu
                loss = loss.cpu()
                y_prim = y_prim.cpu()
                y = y.cpu()
                x = x.cpu()
                np_x = x.detach().numpy()
                np_y_prim = y_prim.data.numpy()
                np_y = y.data.numpy()

                # For OHE
                # idx_y = np.argmax(np_y, axis=1)
                idx_y = np_y
                idx_y_prim = np.argmax(np_y_prim, axis=1)

                acc = np.average((idx_y == idx_y_prim) * 1.0)

                metrics_epoch[f'{stage}_loss'].append(loss.item())  # Tensor(0.1) => 0.1f
                metrics_epoch[f'{stage}_acc'].append(acc.item())

            # Early stopping and keeping track of best metrics
            if stage == 'test':
                test_acc = np.mean(metrics_epoch['test_acc'])
                test_loss = np.mean(metrics_epoch['test_loss'])

                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    elapsed = 0
                elif test_acc <= best_test_acc and elapsed < patience:
                    elapsed += 1
                    pass

                if epoch == 0:
                    best_test_loss = test_loss
                elif test_loss < best_test_loss:
                    best_test_loss = test_loss

                    # Saves model if a better test loss is achieved
                    torch.save(model.cpu().state_dict(), f'./results/{args.sequence_name}/'
                                                         f'{args.run_name}/{args.run_name}-model-{epoch}.pt')
                    model = model.to('cuda')

                metrics['best_test_loss'] = best_test_loss
                metrics['best_test_acc'] = best_test_acc

                metrics_epoch[f'best_{stage}_loss'] = best_test_loss.item()  # Tensor(0.1) => 0.1f
                metrics_epoch[f'best_{stage}_acc'] = best_test_acc.item()

            metrics_strs = []

            for key in metrics_epoch.keys():
                if stage in key:
                    value = np.mean(metrics_epoch[key])
                    metrics[key] = value
                    metrics_strs.append(f'{key}: {round(value, 2)}')

            print(f'epoch: {epoch} {" ".join(metrics_strs)}')

        if elapsed == patience:
            print("Early stop run, no increase in test accuracy")
            elapsed = 0
            max_test_acc = 0
            max_test_loss = 0
            break

        CsvUtils2.add_hparams(
            path_sequence,
            args.run_name,
            args.__dict__,
            metrics_dict=metrics,
            global_step=epoch
        )
