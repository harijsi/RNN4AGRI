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
from sklearn.metrics import classification_report
from torch.autograd import Variable
from csv_utils_2 import CsvUtils2
from file_utils import FileUtils
from datetime import datetime
import os
from sklearn.model_selection import ParameterGrid
import torch_optimizer as optim
import csv

param_grid = {'-learning_rate': [1e-3, 1e-4], '-batch_size': [32, 64, 128, 256],
              '-hidden_size': [32, 64, 128, 256], '-num_layers': [2]}

g_id = 0
num_of_classes = 15
input_size = 16
sequence_name = "BiLSTM_S1S2_15class_balanced_patience50"

# Keeps track of used combinations
try:
    seq_csv = pd.read_csv("results/"+sequence_name+"_params.csv")
    used_combos = []
    for i, row in seq_csv.iterrows():
        lr = row['learning_rate']
        bs = row['batch_size']
        hs = row['hidden_size']
        nl = row['num_layers']
        used_combos.append({'learning_rate': lr, 'batch_size': bs, 'hidden_size': hs, 'num_layers': nl})
except:
    pass

# Specify dataset
print('loading dataset...')
df = pd.read_pickle('../datasets/MDB210921/MDB21092021_cached.pkl').fillna(value=0, axis=0)
df2 = df.loc[~df['Class_ID'].isin([999, 99, 20, 7, 8])].reset_index(drop=True)
print('loading done, proceeding')

# Class numbers to idx
class2idx = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    9: 6,
    11: 7,
    12: 8,
    13: 9,
    14: 10,
    15: 11,
    17: 12,
    18: 13,
    19: 14,
    20: 15
}

# Idx to labels
idx2lab = {
    0: 'VKV',
    1: 'ZKV',
    2: 'RUD',
    3: 'VMIE',
    4: 'ZMIE',
    5: 'AUZ',
    6: 'GRI',
    7: 'VRA',
    8: 'ZRA',
    9: 'PAK',
    10: 'PAP',
    11: 'STAD',
    12: 'ZAL',
    13: 'KUK',
    14: 'KAR',
    15: 'DAZ'
}

df2['Class_ID'].replace(class2idx, inplace=True)
classes = np.unique(df2['Class_ID'])

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

    cur_para_dict = {'learning_rate': g_learning_rate, 'batch_size': g_batch_size,
                     'hidden_size': g_hidden_size, 'num_layers': g_num_layers}

    try:
        if cur_para_dict in used_combos:
            print('Parameter combination already processed, passing...')
            continue
    except:
        pass

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('-id', default=g_id, type=int)
    parser.add_argument('-run_name', default=f'run_{g_id}', type=str)
    parser.add_argument('-sequence_name', default=sequence_name, type=str)
    parser.add_argument('-learning_rate', default=g_learning_rate, type=float)
    parser.add_argument('-batch_size', default=g_batch_size, type=int)
    parser.add_argument('-epochs', default=200, type=int)
    parser.add_argument('-is_cuda', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-sequence_len', default=21, type=int)
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

    # Calculate class distribution
    total_count = len(df2.index) / args.sequence_len
    w_array = np.empty(num_of_classes)
    w_list = []
    w_dict = {}

    # Calculate weight for each class and append to list
    for n in range(0, num_of_classes):
        class_count = len(df2[df2['Class_ID'] == n].index) / args.sequence_len
        weight = (1 / class_count) * total_count / 2
        w_list.append(weight)
        w_dict[str(n)] = class_count

    # Convert list to Torch Tensor
    class_weights = torch.Tensor(w_list).to('cuda')


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
            x1 = (self.data_x[(idx * args.sequence_len):((idx * args.sequence_len) + args.sequence_len)]).values
            x = np.array(x1, dtype="float32")
            x = self.scaler.transform(x)
            x = torch.from_numpy(x)
            y = int(self.data_y['Class_ID'][idx * args.sequence_len])
            y = torch.tensor([int(y)])
            return x, y


    # Define parameters for train-test split
    torch.manual_seed(1)
    dataset_len = len(df2) / args.sequence_len
    train_size = int(0.8 * dataset_len)
    test_size = int(dataset_len - train_size)

    # Create train & test datasets through random split
    train_dataset, test_dataset = torch.utils.data.random_split(
        DatasetTabular(is_train=True, sequence_len=args.sequence_len),
        [train_size, test_size])

    # # Create a balanced testing dataset
    # test_idx_dict = {}
    # test_idxes = []
    # leftover_idxes = []
    #
    # for i in range(0, len(test_dataset)):
    #     class_id = test_dataset[i][1].item()
    #     if class_id not in test_idx_dict.keys():
    #         test_idx_dict[class_id] = 0
    #         test_idxes.append(i)
    #     elif class_id in test_idx_dict.keys() and test_idx_dict[class_id] < 300:
    #         test_idx_dict[class_id] += 1
    #         test_idxes.append(i)
    #     else:
    #         leftover_idxes.append(i)
    #
    # balanced_test_dataset = torch.utils.data.Subset(test_dataset, test_idxes)
    # leftover_test_dataset = torch.utils.data.Subset(test_dataset, leftover_idxes)
    # train_dataset = torch.utils.data.ConcatDataset([train_dataset, leftover_test_dataset])

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
                num_layers=num_layers,
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


    model = ModelBiLSTM(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.num_layers,
                        output_size=num_of_classes)

    loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)
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

        # Initialize empty dataframes for test and train confusion matrices
        test_df = pd.DataFrame(columns=['True', 'Predicted'])
        train_df = pd.DataFrame(columns=['True', 'Predicted'])

        for data_loader in [data_loader_train, data_loader_test]:
            metrics_epoch = {key: [] for key in metrics.keys()}

            stage = 'train'
            if data_loader == data_loader_test:
                stage = 'test'

            for x, y in data_loader:

                y = y.squeeze(1)

                # if cuda move data to cuda device
                if args.is_cuda:
                    x = x.cuda()
                    y = y.cuda()

                y_prim = model.forward(x)

                idxes = torch.arange(0, args.batch_size).to('cuda')
                # loss = loss_func(y_prim, y)

                loss = -torch.mean(class_weights[y[idxes]] * torch.log(y_prim[idxes, y[idxes]] + 1e-20))
                # loss_torch = loss_func(y_prim, y)
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
                np_x = x.numpy()
                np_y_prim = y_prim.data.numpy()
                np_y = y.data.numpy()

                # For OHE
                # idx_y = np.argmax(np_y, axis=1)
                idx_y = np_y
                idx_y_prim = np.argmax(np_y_prim, axis=1)

                acc = np.average((idx_y == idx_y_prim) * 1.0)

                metrics_epoch[f'{stage}_loss'].append(loss.item())  # Tensor(0.1) => 0.1f
                metrics_epoch[f'{stage}_acc'].append(acc.item())

                # This section appends y and y prim idxes to list for calculation of crosstab / accuracy report
                if stage == 'test':
                    df_to_append = pd.DataFrame(columns=['True', 'Predicted'])
                    df_to_append['True'] = idx_y
                    df_to_append['Predicted'] = idx_y_prim
                    test_df = test_df.append(df_to_append)
                elif stage == 'train':
                    df_to_append = pd.DataFrame(columns=['True', 'Predicted'])
                    df_to_append['True'] = idx_y
                    df_to_append['Predicted'] = idx_y_prim
                    train_df = train_df.append(df_to_append)

            # Early stopping and keeping track of best metrics
            if stage == 'test':

                test_acc = np.mean(metrics_epoch['test_acc'])
                test_loss = np.mean(metrics_epoch['test_loss'])

                # if current test accuracy > best test accuracy, reset patience counter and save accuracy metrics
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    elapsed = 0

                    # Replace result class idx with class names
                    test_df['True'].replace(idx2lab, inplace=True)
                    test_df['Predicted'].replace(idx2lab, inplace=True)
                    train_df['True'].replace(idx2lab, inplace=True)
                    train_df['Predicted'].replace(idx2lab, inplace=True)

                    # Exports confusion matrix and accuracy report for test and train sets
                    matrix_test = pd.crosstab(test_df['True'], test_df['Predicted'])
                    matrix_test.to_csv(f'./results/{args.sequence_name}/'
                                       f'{args.run_name}/test_matrix_run_{g_id}_epoch_{epoch}.csv')

                    report_test = classification_report(test_df['True'].values, test_df['Predicted'].values, output_dict=True)
                    report_test_df = pd.DataFrame(report_test).transpose()
                    report_test_df.to_csv(f'./results/{args.sequence_name}/'
                                          f'{args.run_name}/test_report_run_{g_id}_epoch_{epoch}.csv')

                    matrix_train = pd.crosstab(train_df['True'], train_df['Predicted'])
                    matrix_train.to_csv(f'./results/{args.sequence_name}/'
                                        f'{args.run_name}/train_matrix_run_{g_id}_epoch_{epoch}.csv')

                    report_train = classification_report(train_df['True'].values, train_df['Predicted'].values, output_dict=True)
                    report_train_df = pd.DataFrame(report_train).transpose()
                    report_train_df.to_csv(f'./results/{args.sequence_name}/'
                                           f'{args.run_name}/train_report_run_{g_id}_epoch_{epoch}.csv')

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

    if os.path.exists("results/"+sequence_name+"_params.csv"):
        f = open("results/"+sequence_name+"_params.csv", 'a', newline='')
        fieldnames = list(cur_para_dict.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(cur_para_dict)
        f.close()
    else:
        f = open("results/"+sequence_name+"_params.csv", 'w', newline='')
        fieldnames = list(cur_para_dict.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(cur_para_dict)
        f.close()
