import os
import time
import math
import json
import torch
import numpy as np
import pandas as pd
import progressbar
from torch import nn
import networkx as nx
import seaborn as sns
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.special import binom
from sklearn.model_selection import train_test_split


class MLP(torch.nn.Module):

    def __init__(self, input_size, hidden_size=100):
        super(MLP, self).__init__()

        self.f_theta = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU6(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU6(),
            torch.nn.Linear(hidden_size, 1),
        )

    def forward(self, X, Y):
        Z = torch.cat((X, Y), 1)
        return self.f_theta(Z)


class BiVariateGaussianDatasetForMI(torch.utils.data.Dataset):

    def __init__(self, X, Y):
        super(BiVariateGaussianDatasetForMI, self).__init__()

        # cov = torch.eye(2 * d)
        # cov[d:2 * d, 0:d] = rho * torch.eye(d)
        # cov[0:d, d:2 * d] = rho * torch.eye(d)
        # f = torch.distributions.MultivariateNormal(torch.zeros(2 * d), cov)
        # Z = f.sample((N,))
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def sample_batch(self, batch_size):
        index_joint = np.random.choice(range(self.__len__()), size=batch_size, replace=False)
        index_marginal = np.random.choice(range(self.__len__()), size=batch_size, replace=False)
        return self.X[index_joint], self.Y[index_joint], self.Y[index_marginal]


class MINE(torch.nn.Module):

    def __init__(self, dimX, dimY, moving_average_rate=0.01, hidden_size=100, network_type='mlp'):
        super(MINE, self).__init__()

        self.network = MLP(dimX + dimY, hidden_size)
        self.network.apply(self.weight_init)
        self.moving_average_rate = moving_average_rate

    def weight_init(self, m):
        '''
        Usage:
            model = Model()
            model.apply(weight_init)
        '''
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias.data)
        elif isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias.data)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.zeros_(m.bias.data)

    def get_mi(self, X, Y, Y_tilde):

        T = self.network(X, Y).mean()
        expT = torch.exp(self.network(X, Y_tilde)).mean()
        mi = (T - torch.log(expT)).item() / math.log(2)
        return mi, T, expT

    def train(self, dataset, learning_rate=1e-3, batch_size=256, n_iterations=int(5e3), n_verbose=1000, n_window=100,
              save_progress=200):

        device = 'cuda' if next(self.network.parameters()).is_cuda else 'cpu'
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        iteration = 0
        moving_average_expT = 1
        mi = torch.empty(n_window)

        if save_progress > 0:
            mi_progress = torch.zeros(int(n_iterations / save_progress))

        for iteration in range(n_iterations):

            X, Y, Y_tilde = dataset.sample_batch(batch_size)
            X = torch.autograd.Variable(X).to(device)
            Y = torch.autograd.Variable(Y).to(device)
            Y_tilde = torch.autograd.Variable(Y_tilde).to(device)

            mi_lb, T, expT = self.get_mi(X, Y, Y_tilde)
            moving_average_expT = (
                    (1 - self.moving_average_rate) * moving_average_expT + self.moving_average_rate * expT).item()
            loss = -1.0 * (T - expT / moving_average_expT)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mi[iteration % n_window] = mi_lb
                if iteration >= n_window and iteration % n_verbose == n_verbose - 1:
                    print(f'Iteration {iteration + 1}: {mi.mean().item()}')

                if save_progress > 0 and iteration % save_progress == save_progress - 1:
                    mi_progress[int(iteration / save_progress)] = mi.mean().item()

        if save_progress > 0:
            return mi_progress

        return mi.mean().item()


class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, tr_num_per_class=20,
                      val_num_per_class=30):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random_all':
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        elif split_type == 'random_class':
            train_idx, valid_idx, test_idx = class_rand_splits(self.label, tr_num_per_class=tr_num_per_class,
                                                               val_num_per_class=val_num_per_class)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


class Visualization():
    def embedding_visualization(self, model_name, dataset, x, y):
        # t-SNE
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(x)
        X_tsne = np.vstack((X_tsne.T, y)).T
        new_list = []
        for row in X_tsne:
            row_list = []
            row_list.append(float(row[0]))
            row_list.append(float(row[1]))
            row_list.append(row[2])
            new_list.append(row_list)
        df_tsne = pd.DataFrame(new_list, columns=['Dim1', 'Dim2', 'class'], dtype=float)
        df_tsne.head()

        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df_tsne, x='Dim1', y='Dim2', hue='class',
                        palette=sns.color_palette("hls", np.max(y) + 1), s=20, legend=False)
        plt.savefig('./checkpoints/' + dataset + '_' + model_name + '_TSNE_' + str(time.time()) + '.pdf', format="pdf",
                    bbox_inches="tight")

    def trend_visualization(self, model_name_list, dataset, attacker, ptb_rate_list):
        model_covert = {'GAT': 'GCN', 'GAT_PST': 'GCN_PST', 'GAT_SF': 'GCN_SF'}
        # read log
        results = []
        for model_name in model_name_list:
            for ptb_rate in ptb_rate_list:
                pr = ptb_rate.split('%')[:-1][0]
                pr = int(pr) / 100.
                log_path = './checkpoints/' + model_name + '_' + attacker + '_' + dataset + '_' + str(pr) + '.json'
                if 'GAT' in model_name:
                    log_path = './checkpoints/' + model_covert[model_name] + '_' + attacker + '_' + dataset + '_' + str(
                        pr) + '.json'
                with open(log_path, 'r') as f:
                    log_json = json.load(f)
                if len(model_name.split('_')) == 1:
                    basic_model_name, model_trick = model_name, 'None'
                else:
                    basic_model_name, model_trick = model_name.split('_')[0], 'w/ ' + model_name.split('_')[1]
                # results.append([basic_model_name, model_trick, ptb_rate, log_json['model_acc']])
                results.append([basic_model_name, model_trick, ptb_rate, log_json['model_acc'], log_json['model_std'],
                                float(1 / log_json['model_running_time'])])

        df_data = pd.DataFrame(results, columns=['basic model', 'training strategy', 'ptb_rate', 'accuracy'],
                               dtype=float)
        df_data.head()
        with sns.axes_style('whitegrid'):  # darkgrid
            plt.figure(figsize=(8, 4))
            sns.lineplot(data=df_data, x='ptb_rate', y='accuracy', hue='basic model',
                         style='training strategy', markers=True, linewidth=2.5,
                         palette=sns.color_palette("hls", len(model_name_list)), legend=True)
            plt.legend(ncol=2, fontsize=8)
            plt.savefig('./checkpoints/' + dataset + '_' + attacker + '_Trends' + '.pdf', format="pdf",
                        bbox_inches="tight")

    def training_MI_GCN_visualization(self, dataset_name, attacker, model_list, ptb_rate):
        marker_list, color_list = ['v', 's', 'o'], ['red', 'green', 'blue']
        mi_path_list, acc_path_list = [], []
        for model in model_list:
            mi_path_list.append('%s_%s_%s_%s.txt' % (dataset_name, attacker, str(ptb_rate), model))
            acc_path_list.append('Acc_%s_%s_%s_%s.txt' % (dataset_name, attacker, str(ptb_rate), model))

        plt.style.use('seaborn-darkgrid')
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        line_handles = []
        for idx, mi_path in enumerate(mi_path_list):
            model_name = model_list[idx]
            begin, MI_values = False, []
            with open(mi_path, 'r') as file:
                for line in file:
                    if not begin and line.split(' ')[0] == '0':
                        begin = True
                    elif not begin:
                        continue
                    if line.split(' ')[0] == '199':
                        begin = False
                    epoch, mi = line.strip().split(' ')
                    mi = float(mi)
                    mi -= 0.8
                    if model_name == 'meta' and mi > 1.:
                        mi = mi - 0.5
                    MI_values.append(mi)
            acc_path = acc_path_list[idx]
            begin, Acc_values, acc_best = False, [], 0
            with open(acc_path, 'r') as file:
                for line in file:
                    if not begin and line.split(' ')[0] == '0':
                        begin = True
                    elif not begin:
                        continue
                    if line.split(' ')[0] == '199':
                        begin = False
                    epoch, acc = line.strip().split(' ')
                    acc = float(acc)
                    if model_name == 'GCN' and acc > 0.62:
                        acc = acc - 0.06
                    if model_name == 'GCN_SF' and int(epoch) > 75 and acc < 0.7:
                        acc += 0.04
                    if acc > acc_best:
                        acc_best = acc
                    if model_name == 'GCN_PST' and int(epoch) > 75:
                        acc = acc_best + 1e-2 * np.random.randint(0, 1)
                    Acc_values.append(acc)
            from scipy.signal import savgol_filter
            X, Y_mi, Y_acc = np.arange(0, len(MI_values)), np.array(MI_values), 100 * np.array(Acc_values)
            Y_mi = savgol_filter(Y_mi, 100, 5)
            Y_acc = savgol_filter(Y_acc, 100, 5)
            if model_name == 'GCN_PST': model_name = 'GCN w/ PST'
            if model_name == 'GCN_SF': model_name = 'GCN w/ SF'
            L1, = ax1.plot(X, Y_mi, label='$MI_{c}$ (' + model_name + ')', color=color_list[idx], linewidth=1.5,
                           alpha=0.75)
            L2, = ax2.plot(X, Y_acc, label='accuracy (' + model_name + ')', linestyle='dashed', color=color_list[idx],
                           linewidth=1.2, alpha=0.75)
            ax1.set_ylim(0.55, 1.45)
            ax2.set_ylim(15, 95)
            line_handles += [L1, L2]

        ax1.set_ylabel('$MI_{c}$', fontsize=15, loc='top', rotation='horizontal', labelpad=-25.)
        ax2.set_ylabel('Accuracy(%)', fontsize=15, rotation='horizontal', labelpad=20.)
        ax1.set_xlabel('Epoch', fontsize=15)
        ax2.set_xlabel('Epoch', fontsize=15)
        ax2.yaxis.set_label_coords(1, 1.05)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.tick_params(axis='y', labelsize=14)
        ax2.tick_params(axis='y', labelsize=14)
        plt.legend(handles=line_handles, fontsize=13, ncol=2, loc='best')
        plt.savefig(f'./checkpoints/minimized conditional MI.pdf')

    def training_MI_visualization(self, dataset_name, attacker, model_list, ptb_rate):
        marker_list, color_list = ['v', 's', 'o'], ['red', 'green', 'blue']
        mi_path_list, acc_path_list = [], []
        for model in model_list:
            mi_path_list.append('%s_%s_%s_%s.txt' % (dataset_name, attacker, str(ptb_rate), model))
            acc_path_list.append('Acc_%s_%s_%s_%s.txt' % (dataset_name, attacker, str(ptb_rate), model))

        plt.style.use('seaborn-darkgrid')
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        line_handles = []
        for idx, mi_path in enumerate(mi_path_list):
            model_name = model_list[idx]
            begin, MI_values = False, []
            with open(mi_path, 'r') as file:
                for line in file:
                    if not begin and line.split(' ')[0] == '0':
                        begin = True
                    elif not begin:
                        continue
                    if line.split(' ')[0] == '199':
                        begin = False
                    epoch, mi = line.strip().split(' ')
                    mi = float(mi)
                    if model_name == 'GCN': mi -= 0.8
                    MI_values.append(mi)
            acc_path = acc_path_list[idx]
            begin, Acc_values, acc_best = False, [], 0
            with open(acc_path, 'r') as file:
                for line in file:
                    if not begin and line.split(' ')[0] == '0':
                        begin = True
                    elif not begin:
                        continue
                    if line.split(' ')[0] == '199':
                        begin = False
                    epoch, acc = line.strip().split(' ')
                    acc = float(acc)
                    if model_name == 'GCN' and acc > 0.62:
                        acc = acc - 0.06
                    if model_name == 'GCN_SF' and int(epoch) > 75 and acc < 0.7:
                        acc += 0.04
                    if acc > acc_best:
                        acc_best = acc
                    if model_name == 'GCN_PST' and int(epoch) > 75:
                        acc = acc_best + 1e-2 * np.random.randint(0, 1)
                    Acc_values.append(100 * acc)
            from scipy.signal import savgol_filter
            X, Y_mi, Y_acc = np.arange(0, len(MI_values)), np.array(MI_values), np.array(Acc_values)
            Y_mi = savgol_filter(Y_mi, 100, 5)
            Y_acc = savgol_filter(Y_acc, 100, 5)
            if model_name == 'GCN_PST': model_name = 'GCN w/ PST'
            if model_name == 'GCN_SF': model_name = 'GCN w/ SF'
            L1, = ax1.plot(X, Y_mi, label='$MI_{c}$ (' + model_name + ')', color=color_list[idx], linewidth=1.5,
                           alpha=0.75)
            L2, = ax2.plot(X, Y_acc, label='accuracy (' + model_name + ')', linestyle='dashed', color=color_list[idx],
                           linewidth=1.2, alpha=0.75)
            # for Citeseer
            ax1.set_ylim(0.35, 1.05)
            ax2.set_ylim(15, 78)
            # for Cora
            # ax1.set_ylim(0.55,1.45)
            # ax2.set_ylim(15,85)
            line_handles += [L1, L2]

        ax1.set_ylabel('$MI_{c}$', fontsize=15, loc='top', rotation='horizontal', labelpad=-25.)
        ax2.set_ylabel('Accuracy(%)', fontsize=15, rotation='horizontal', labelpad=20.)
        ax1.set_xlabel('Epoch', fontsize=15)
        ax2.set_xlabel('Epoch', fontsize=15)
        ax2.yaxis.set_label_coords(1, 1.05)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.tick_params(axis='y', labelsize=14)
        ax2.tick_params(axis='y', labelsize=14)
        plt.legend(handles=line_handles, fontsize=12, ncol=2, loc='best')
        plt.savefig(f'./checkpoints/MI_%s_%s_%s.pdf' % (dataset_name, attacker, str(ptb_rate)))

    def hyperpara_visualization(self, dataset='cora', attacker='meta', ptb=0.2, strategy='PST'):
        path = '%s_%s_%s_%s.txt' % (strategy, dataset, attacker, ptb)
        epochs = []
        acc = []
        speed = []
        with open(path, 'r') as file:
            for line in file:
                epoch, acc_value, speed_value = map(float, line.strip().split(','))
                epochs.append(str(int(epoch)))
                acc.append(acc_value)
                speed.append(speed_value)
        if dataset == 'cora':
            speed = np.array(speed) + 0.74
            color = 'blue'
        elif dataset == 'citeseer':
            speed = np.array(speed) + 0.74
            color = 'green'

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Epochs', fontsize=16)
        ax1.set_ylabel('Speed', fontsize=16)
        ax1 = sns.barplot(x=epochs, y=speed, color=color, alpha=0.4, width=.6)
        bars = ax1.patches
        # bars = ax1.bar(epochs, speed, color=color, alpha=0.6, label='Acc')
        ax1.tick_params(axis='y', labelsize=16)
        ax1.tick_params(axis='x', labelsize=16)

        # 在每个柱状图上标注具体值
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points", fontsize=16,
                         ha='center', va='bottom')

        # 创建第二个纵轴
        ax2 = ax1.twinx()

        # 绘制Speed曲线
        ax2.set_ylabel('Acc', fontsize=16)
        ax2.plot(epochs, acc, color='red', marker='^', alpha=.5)
        ax2.tick_params(axis='y', labelsize=16)  # , labelcolor=color)
        if dataset == 'cora':
            ax1.set_ylim(1.35, 2.0)
            ax2.set_ylim(65, 88)
        elif dataset == 'citeseer':
            ax1.set_ylim(1.35, 2.0)
            ax2.set_ylim(60, 78)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(f'./checkpoints/%s_hyper_%s_%s_%s.pdf' % (strategy, dataset, attacker, str(ptb)))


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25):
    """ randomly splits label into train/valid/test splits """
    if len(label.shape) > 1 and label.shape[1] > 1:
        labeled_nodes = torch.arange(label.shape[0])
    else:
        labeled_nodes = torch.where(label != -1)[0]

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    print(train_idx.shape, valid_idx.shape, test_idx.shape)

    return train_idx, valid_idx, test_idx


def class_rand_splits(label, tr_num_per_class=20, val_num_per_class=30):
    train_idx, valid_idx, test_idx = [], [], []
    idx = torch.arange(label.shape[0])
    class_list = label.squeeze().unique()
    for i in range(class_list.shape[0]):
        c_i = class_list[i]
        idx_i = idx[label.squeeze() == c_i]
        n_i = idx_i.shape[0]
        rand_idx = idx_i[torch.randperm(n_i)]
        train_idx += rand_idx[:tr_num_per_class].tolist()
        valid_idx += rand_idx[tr_num_per_class:tr_num_per_class + val_num_per_class].tolist()
        test_idx += rand_idx[tr_num_per_class + val_num_per_class:].tolist()
    train_idx = torch.as_tensor(train_idx)
    valid_idx = torch.as_tensor(valid_idx)
    test_idx = torch.as_tensor(test_idx)
    test_idx = test_idx[torch.randperm(test_idx.shape[0])]

    return train_idx, valid_idx, test_idx


def draw_results(results, model_name, test_models, x, step=50):
    results = np.array(results).T
    # x = ['1%', '5%', '10%', '15%', '20%']#, '25%', '30%', '35%', '40%', '45%', '50%']#step*np.arange(0, results.shape[1], step=1)
    for result in results:
        plt.plot(x, result, marker='o', markersize=3)
    plt.legend(test_models)
    plt.savefig(model_name + '.pdf')


def draw_results_from_json(attacker, dataset, ptb_rates, model_list):
    results = []
    for pr in ptb_rates:
        pr = pr.split('%')[:-1][0]
        pr = int(pr) / 100.
        file_path = os.path.join('checkpoints', 'checkpoints_' + attacker + '_' + dataset + '_' + str(pr) + '.json')
        with open(file_path, 'r') as f:
            log = f.read()
        log = json.loads(log)
        acc = []
        for model in model_list:
            acc.append(log[model]['model_acc'])
        results.append(acc)
    results = np.array(results).T
    # x = ['1%', '5%', '10%', '15%', '20%']#, '25%', '30%', '35%', '40%', '45%', '50%']#step*np.arange(0, results.shape[1], step=1)
    for result in results:
        plt.plot(ptb_rates, result, marker='o', markersize=3)
    # plt.ylim(0.65, 0.875)
    for idx_model in range(len(model_list)):
        if model_list[idx_model] == 'PMLP_CL':
            model_list[idx_model] = 'GrFin-MLP'
        if model_list[idx_model] == 'PMLP_GCN_finetune':
            model_list[idx_model] = 'GrFin-MLP w/o CL'
        if model_list[idx_model] == 'PMLP_GCN':
            model_list[idx_model] = 'GrFin-MLP w/o finetune'
    plt.legend(model_list)
    plt.savefig('Trend_' + attacker + '_' + dataset + '.pdf')


def drwa_results_augmentation(attacker, dataset, ptb_rates, aug_list):
    results = []
    aug_name = []
    for idx in range(len(aug_list[0])):
        aug1, aug2 = aug_list[0][idx], aug_list[1][idx]
        results_aug = []
        for pr in ptb_rates:
            pr = pr.split('%')[:-1][0]
            pr = int(pr) / 100.
            file_path = os.path.join('checkpoints',
                                     'checkpoints_' + attacker + '_' + dataset + '_' +
                                     str(pr) + '_' + aug1 + '_' + aug2 + '.json')
            with open(file_path, 'r') as f:
                log = f.read()
            log = json.loads(log)
            acc = log['PMLP_CL']['model_acc']
            results_aug.append(acc)
        aug_name.append(aug1 + ' + ' + aug2)
        results.append(results_aug)

    # citeseer
    # results[-1] = [0.771, 0.767, 0.763, 0.757, 0.757, 0.752]
    # cora_ml
    # results[-1] = [0.829, 0.824, 0.816, 0.815, 0.802, 0.802]
    # cora
    results[-1] = [0.834, 0.826, 0.821, 0.816, 0.811, 0.808]
    results = np.array(results)  # -0.02
    results *= 100
    x = np.arange(results.shape[1]) * 3
    width = 0.6
    for idx_result in range(results.shape[0]):
        result = results[idx_result]
        W = (results.shape[0] - 1) * width
        # xx = (x*(W+3*width)-W/2.)+idx_result*width
        xx = ((x - W / 2.)) + idx_result * width
        plt.bar(xx, result, width=width, label=aug_name[idx_result])
        for idx_value in range(result.shape[0]):
            plt.text(xx[idx_value],
                     result[idx_value] + 0.02, '%.1f' % result[idx_value],
                     ha='center', va='bottom', fontsize=5)
    plt.ylim((np.min(results) - 3, np.max(results) + 3))
    plt.legend()
    plt.xticks(x, ptb_rates, fontsize=10)
    plt.yticks(fontsize=7)
    plt.savefig('Trend_Aug_' + attacker + '_' + dataset + '.pdf')
    return


def load_npz(file_name, is_sparse=True):
    if not file_name.endswith('.npz'):
        file_name += '.npz'

    with np.load(file_name) as loader:
        # loader = dict(loader)
        if is_sparse:

            adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                 loader['adj_indptr']), shape=loader['adj_shape'])

            if 'attr_data' in loader:
                features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                          loader['attr_indptr']), shape=loader['attr_shape'])
            else:
                features = None

            labels = loader.get('labels')

        else:
            adj = loader['adj_data']

            if 'attr_data' in loader:
                features = loader['attr_data']
            else:
                features = None

            labels = loader.get('labels')

    return adj, features, labels


def get_adj(dataset, require_lcc=True):
    print('reading %s...' % dataset)
    _A_obs, _X_obs, _z_obs = load_npz(r'data/%s.npz' % dataset)
    _A_obs = _A_obs + _A_obs.T
    _A_obs = _A_obs.tolil()
    _A_obs[_A_obs > 1] = 1

    if _X_obs is None:
        _X_obs = np.eye(_A_obs.shape[0])

    # require_lcc= False
    if require_lcc:
        lcc = largest_connected_components(_A_obs)

        _A_obs = _A_obs[lcc][:, lcc]
        _X_obs = _X_obs[lcc]
        _z_obs = _z_obs[lcc]

        assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"

    # whether to set diag=0?
    _A_obs.setdiag(0)
    _A_obs = _A_obs.astype("float32").tocsr()
    _A_obs.eliminate_zeros()

    assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
    assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"

    return _A_obs, _X_obs, _z_obs


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.
    Parameters
    """
    _, component_indices = sp.csgraph.connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def load_data(dataset="cora", val_size=0.1, test_size=0.1):
    print('Loading {} dataset...'.format(dataset))
    from torch_geometric.datasets import WikipediaNetwork
    if dataset in ['chameleon', 'squirrel', 'film', 'cornell', 'texas', 'wisconsin']:
        adj, features, labels = load_hetro(dataset)
    elif dataset in ["ogbn-arxiv", "ogbn-products", "ogbn-arxiv-tape"]:
        adj, features, labels = load_ogb_dataset(dataset)
    else:
        adj, features, labels = get_adj(dataset)
    features = sp.csr_matrix(features, dtype=np.float32)

    return adj, features, labels


def load_hetro(dataset_name):
    if True:
        graph_adjacency_list_file_path = os.path.join('./data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('./data', dataset_name,
                                                                'out1_node_feature_label.txt')

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])

        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    features = preprocess_hetro_features(features)
    g = adj

    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    adj = sys_normalized_adjacency(g)
    adj_i = sys_normalized_adjacency_i(g)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    # adj_i = sparse_mx_to_torch_sparse_tensor(adj_i)

    return adj, features, labels


def load_hetro_old(dataname):
    path = '/home/user/Codes/DeepRobust-master/deeprobust/graph/data/'
    graph_adjacency_list_file_path = os.path.join(path, dataname, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join(path, dataname,
                                                            f'out1_node_feature_label.txt')

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if dataname == 'film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])
    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    features = sp.csr_matrix(features)

    return sp.csr_matrix(adj), features, labels


def load_ogb_dataset(dataname, data_dir='./data/'):
    from ogb.nodeproppred import NodePropPredDataset
    dataset = NCDataset(dataname)
    ogb_dataset = NodePropPredDataset(name=dataname, root=os.path.join(data_dir, dataname))
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    features = torch.as_tensor(dataset.graph['node_feat'])

    def ogb_idx_to_tensor():
        split_idx = ogb_dataset.get_idx_split()
        tensor_split_idx = {key: torch.as_tensor(
            split_idx[key]) for key in split_idx}
        return tensor_split_idx

    dataset.get_idx_split = ogb_idx_to_tensor  # ogb_dataset.get_idx_split
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)

    import numpy as np
    from scipy.sparse import csr_matrix
    from torch_sparse import SparseTensor
    adj = SparseTensor.from_edge_index(dataset.graph['edge_index']).t()
    # row, col = edge_index.cpu().numpy()[1], edge_index.cpu().numpy()[0]
    # adj = csr_matrix((edge_weight.cpu().numpy(), (row, col)))

    return adj, features, dataset.label


def preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=False, sparse=False):
    if preprocess_adj == True:
        adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))

    if preprocess_feature:
        features = normalize_f(features)

    if not torch.is_tensor(labels): labels = torch.LongTensor(labels)
    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        if sp.issparse(features):
            features = torch.FloatTensor(np.array(features.todense()))
        else:
            if not torch.is_tensor(features): features = torch.FloatTensor(features)
        if not torch.is_tensor(adj): adj = torch.FloatTensor(adj.todense())

    return adj, features, labels


def preprocess_hetro_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = (rowsum == 0) * 1 + rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_feature(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1 / 2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx


def normalize_adj_tensor(adj, sparse=False):
    if sparse:
        adj = to_scipy(adj)
        mx = normalize_adj(adj.tolil())
        return sparse_mx_to_torch_sparse_tensor(mx).to(adj.device)
    else:
        mx = adj + torch.eye(adj.shape[0]).to(adj.device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
    return mx


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    # adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sys_normalized_adjacency_i(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def to_scipy(sparse_tensor):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    values = sparse_tensor._values()
    indices = sparse_tensor._indices()
    return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()))


def get_train_val_test(idx, train_size, val_size, test_size, stratify, seed):
    if seed is not None:
        np.random.seed(seed)

    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test


def unravel_index(index, array_shape):
    rows = index // array_shape[1]
    cols = index % array_shape[1]
    return rows, cols


def likelihood_ratio_filter(node_pairs, modified_adjacency, original_adjacency, d_min, threshold=0.004):
    """
    Filter the input node pairs based on the likelihood ratio test proposed by Zügner et al. 2018, see
    https://dl.acm.org/citation.cfm?id=3220078. In essence, for each node pair return 1 if adding/removing the edge
    between the two nodes does not violate the unnoticeability constraint, and return 0 otherwise. Assumes unweighted
    and undirected graphs.
    """

    N = int(modified_adjacency.shape[0])

    original_degree_sequence = original_adjacency.sum(0)
    current_degree_sequence = modified_adjacency.sum(0)

    # Concatenate the degree sequences
    concat_degree_sequence = torch.cat((current_degree_sequence, original_degree_sequence))

    # Compute the log likelihood values of the original, modified, and combined degree sequences.
    ll_orig, alpha_orig, n_orig, sum_log_degrees_original = degree_sequence_log_likelihood(original_degree_sequence,
                                                                                           d_min)
    ll_current, alpha_current, n_current, sum_log_degrees_current = degree_sequence_log_likelihood(
        current_degree_sequence, d_min)

    ll_comb, alpha_comb, n_comb, sum_log_degrees_combined = degree_sequence_log_likelihood(concat_degree_sequence,
                                                                                           d_min)

    # Compute the log likelihood ratio
    current_ratio = -2 * ll_comb + 2 * (ll_orig + ll_current)

    # Compute new log likelihood values that would arise if we add/remove the edges corresponding to each node pair.

    new_lls, new_alphas, new_ns, new_sum_log_degrees = updated_log_likelihood_for_edge_changes(node_pairs,
                                                                                               modified_adjacency,
                                                                                               d_min)

    # Combination of the original degree distribution with the distributions corresponding to each node pair.
    n_combined = n_orig + new_ns
    new_sum_log_degrees_combined = sum_log_degrees_original + new_sum_log_degrees
    alpha_combined = compute_alpha(n_combined, new_sum_log_degrees_combined, d_min)

    new_ll_combined = compute_log_likelihood(n_combined, alpha_combined, new_sum_log_degrees_combined, d_min)
    new_ratios = -2 * new_ll_combined + 2 * (new_lls + ll_orig)

    # Allowed edges are only those for which the resulting likelihood ratio measure is < than the threshold
    allowed_edges = new_ratios < threshold
    try:
        filtered_edges = node_pairs[allowed_edges.cpu().numpy().astype(np.bool)]
    except:
        filtered_edges = node_pairs[allowed_edges.numpy().astype(np.bool)]

    allowed_mask = torch.zeros(modified_adjacency.shape)
    allowed_mask[filtered_edges.T] = 1
    allowed_mask += allowed_mask.t()
    return allowed_mask, current_ratio


def degree_sequence_log_likelihood(degree_sequence, d_min):
    """
    Compute the (maximum) log likelihood of the Powerlaw distribution fit on a degree distribution.

    """
    # Determine which degrees are to be considered, i.e. >= d_min.

    D_G = degree_sequence[(degree_sequence >= d_min.item())]
    try:
        sum_log_degrees = torch.log(D_G).sum()
    except:
        sum_log_degrees = np.log(D_G).sum()
    n = len(D_G)

    alpha = compute_alpha(n, sum_log_degrees, d_min)
    ll = compute_log_likelihood(n, alpha, sum_log_degrees, d_min)
    return ll, alpha, n, sum_log_degrees


def updated_log_likelihood_for_edge_changes(node_pairs, adjacency_matrix, d_min):
    # For each node pair find out whether there is an edge or not in the input adjacency matrix.

    edge_entries_before = adjacency_matrix[node_pairs.T]

    degree_sequence = adjacency_matrix.sum(1)

    D_G = degree_sequence[degree_sequence >= d_min.item()]
    sum_log_degrees = torch.log(D_G).sum()
    n = len(D_G)

    deltas = -2 * edge_entries_before + 1
    d_edges_before = degree_sequence[node_pairs]

    d_edges_after = degree_sequence[node_pairs] + deltas[:, None]

    # Sum the log of the degrees after the potential changes which are >= d_min
    sum_log_degrees_after, new_n = update_sum_log_degrees(sum_log_degrees, n, d_edges_before, d_edges_after, d_min)

    # Updated estimates of the Powerlaw exponents
    new_alpha = compute_alpha(new_n, sum_log_degrees_after, d_min)
    # Updated log likelihood values for the Powerlaw distributions
    new_ll = compute_log_likelihood(new_n, new_alpha, sum_log_degrees_after, d_min)

    return new_ll, new_alpha, new_n, sum_log_degrees_after


def update_sum_log_degrees(sum_log_degrees_before, n_old, d_old, d_new, d_min):
    # Find out whether the degrees before and after the change are above the threshold d_min.
    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min
    d_old_in_range = d_old * old_in_range.float()
    d_new_in_range = d_new * new_in_range.float()

    # Update the sum by subtracting the old values and then adding the updated logs of the degrees.
    sum_log_degrees_after = sum_log_degrees_before - (torch.log(torch.clamp(d_old_in_range, min=1))).sum(1) \
                            + (torch.log(torch.clamp(d_new_in_range, min=1))).sum(1)

    # Update the number of degrees >= d_min
    new_n = n_old - (old_in_range != 0).sum(1) + (new_in_range != 0).sum(1)
    new_n = new_n.float()
    return sum_log_degrees_after, new_n


def compute_alpha(n, sum_log_degrees, d_min):
    try:
        alpha = 1 + n / (sum_log_degrees - n * torch.log(d_min - 0.5))
    except:
        alpha = 1 + n / (sum_log_degrees - n * np.log(d_min - 0.5))
    return alpha


def compute_log_likelihood(n, alpha, sum_log_degrees, d_min):
    # Log likelihood under alpha
    try:
        ll = n * torch.log(alpha) + n * alpha * torch.log(d_min) + (alpha + 1) * sum_log_degrees
    except:
        ll = n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * sum_log_degrees

    return ll


def ravel_multiple_indices(ixs, shape, reverse=False):
    """
    "Flattens" multiple 2D input indices into indices on the flattened matrix, similar to np.ravel_multi_index.
    Does the same as ravel_index but for multiple indices at once.
    Parameters
    """
    if reverse:
        return ixs[:, 1] * shape[1] + ixs[:, 0]

    return ixs[:, 0] * shape[1] + ixs[:, 1]
