from torch import optim
import torch.nn.functional as F
from conv import *
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import GCL.losses as L
from GCL.models import *
from utils import *
import utils
import time
import GCL.augmentors as Aug
from models import *
from deeprobust.graph.defense import *


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, num_layers=2, dropout=0.5, num_node=0):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.ff_bias = True  # Use bias for FF layers in default

        self.bns = nn.BatchNorm1d(hidden_channels, affine=False, track_running_stats=False)
        self.activation = F.relu

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(
            nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias))  # 1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias))  # 1
        self.reset_parameters()

    def reset_parameters(self):
        for mlp in self.fcs:
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_mp=True):
        for i in range(self.num_layers - 1):
            x = x @ self.fcs[i].weight.t()
            if use_mp: x = gcn_conv(x, edge_index)
            if self.ff_bias: x = x + self.fcs[i].bias
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = x @ self.fcs[-1].weight.t()
        if use_mp: x = gcn_conv(x, edge_index)
        if self.ff_bias: x = x + self.fcs[-1].bias
        return x


class BasicGNN(nn.Module):
    def __init__(self, args, nnodes, nfeat, nclass, nhid, dropout,
                 device, finetune_epoch=3, use_mp=False,
                 basic_model='GNN', trick='SF'):
        super(BasicGNN, self).__init__()
        self.lr = args.lr
        self.epochs = args.epochs
        self.device = device
        self.weight_decay = args.weight_decay
        if basic_model == 'SGC':
            self.model = SGC(nfeat, nhid, nclass).to(device)
        elif basic_model == 'APPNP':
            self.model = APPNP(nfeat, nhid, nclass).to(device)
        elif basic_model == 'GCNII':
            self.model = GCNII(nfeat, nhid, nclass).to(device)
        elif basic_model == 'NoisyGCN':
            self.model = MyNoisyGCN(nfeat, nhid, nclass).to(device)
        elif basic_model == 'GCN-Jacard':
            self.model = GNN(nfeat, nhid, nclass).to(device)
            self.gcn_jaccard = GCNJaccard(nfeat=nfeat, nclass=nclass,
                                          nhid=nhid, dropout=dropout)
        elif basic_model == 'SimP-GCN':
            self.model = SimPGCN(nnodes=nnodes, nfeat=nfeat, nclass=nclass,
                                 nhid=nhid, dropout=dropout, device=device)
        else:
            self.model = GNN(nfeat, nhid, nclass).to(device)
        self.basic_model = basic_model
        self.trick = trick
        self.finetune_epoch = finetune_epoch
        self.um = use_mp

    def X_aug(self, features, edge_index, idx_train, ratio=-1):
        if ratio <= 0:
            return True, torch.ones_like(features).to(features.device), edge_index
        elif ratio == 1:
            return True, features, edge_index
        augor = Aug.NodeDropping(pn=ratio)
        new_features, new_edge_index, edge_weight = augor(features, edge_index)
        return True, new_features, new_edge_index

    def A_aug(self, features, edge_index, idx_train, ratio=-1):
        if ratio <= 0:
            return False, features, edge_index
        elif ratio == 1:
            return True, features, edge_index
        edge_num = edge_index.shape[1]
        edge_choiced = np.random.choice(np.arange(0, edge_num), int(ratio * edge_num), replace=False)
        edge_index = edge_index.T[edge_choiced].T
        return True, features, edge_index

    def pst(self, epoch, features, edge_index, idx_train):
        stages = {0: 0.25, self.epochs - 150: 0.5, self.epochs - 100: 0.75, self.epochs - 50: 1.0}
        try:
            ratio = stages[epoch]
            um, features_r, edge_index_r = self.A_aug(features, edge_index, idx_train, ratio)
        except:
            um, features_r, edge_index_r = False, features, edge_index
        return um, features_r, edge_index_r

    def fit(self, features, mod_adj, labels, idx_train, idx_val):
        loss_cl = 0
        y = labels.detach().cpu().numpy()
        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=False).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        edge_index = mod_adj.nonzero().t().contiguous()
        if self.basic_model == 'GCN-Jacard':
            self.gcn_jaccard.threshold = 0.01
            adj_csr = mod_adj.clone().detach().cpu().to_sparse()
            row, col, data, shape = adj_csr._indices()[0], adj_csr._indices()[1], adj_csr._values(), adj_csr.size()
            adj_csr = sp.csr_matrix((data, (row, col)), shape=shape)
            features_csr = features.clone().detach().cpu().to_sparse()
            row, col, data, shape = features_csr._indices()[0], features_csr._indices()[
                1], features_csr._values(), features_csr.size()
            features_csr = sp.csr_matrix((data, (row, col)), shape=shape)
            adj_csr = self.gcn_jaccard.drop_dissimilar_edges(features_csr, adj_csr)
            mod_adj_p = torch.FloatTensor(adj_csr.todense()).to(mod_adj.device)
        # edge_index = mod_adj_p.nonzero().t().contiguous()

        if self.trick == 'SF':
            x, x1, x2 = features.clone().detach().cpu().numpy(), features.clone().detach().cpu().numpy(), \
                features.clone().detach().cpu().numpy()
            sampled_train_idx = np.random.choice(idx_train, int(idx_train.shape[0]), replace=False)
            for sti in sampled_train_idx:
                num = torch.nonzero(mod_adj[sti]).size()[0]
                sameclass_nodes = np.where(y[idx_train] == y[sti])[0]
                sameclass_id = np.random.choice(idx_train[sameclass_nodes], num, replace=True)
                x1[sti] = (x[sti] + np.mean(x[sameclass_id], axis=0)) / 2
                unclass_nodes = np.where(y[idx_train] != y[sti])[0]
                unclass_id = np.random.choice(idx_train[unclass_nodes], num, replace=True)
                x2[sti] = (x[sti] + np.mean(x[unclass_id], axis=0)) / 2
            x1, x2 = torch.Tensor(x1).to(self.device), torch.Tensor(x2).to(self.device)

        # training
        self.model.train()
        um, features_r, edge_index_r = self.um, features, edge_index
        MI_values = []
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            if self.trick == 'SF' and epoch > self.epochs - self.finetune_epoch:
                output1, output2 = self.model(x1, edge_index, use_mp=True), self.model(x2, edge_index, use_mp=True)
                loss_cl = contrast_model(output1, output2)
            if self.trick == 'PST':
                um, features_r, edge_index_r = self.pst(epoch, features, edge_index, idx_train)

            output = self.model(features_r, edge_index_r, use_mp=um)
            output_log = F.log_softmax(output, dim=1)
            loss_train = F.nll_loss(output_log[idx_train], labels[idx_train])
            loss_train += 1. * loss_cl
            loss_train.backward()
            optimizer.step()
        acc_train = utils.accuracy(output[idx_train], labels[idx_train])
        return acc_train.item()

    def test(self, features, mod_adj, labels, idx_test):
        self.model.eval()
        edge_index = mod_adj.nonzero().t().contiguous()
        output = self.model(features, edge_index, True)
        output = F.log_softmax(output, dim=1)
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        return output, acc_test.item()


class ReconGNN(nn.Module):
    def __init__(self, args, nnodes, nfeat, nclass, nhid, dropout,
                 device, finetune_epoch=3, use_mp=False):
        super(ReconGNN, self).__init__()
        self.lr = args.lr
        self.epochs = args.epochs
        self.device = device
        self.weight_decay = args.weight_decay
        self.model = GNN(nfeat, nhid, nclass).to(device)
        self.finetune_epoch = finetune_epoch
        self.um = use_mp

    def fit(self, features, mod_adj, labels, idx_train, idx_val):
        loss_cl, loss_rec = 0, 0
        y = labels.detach().cpu().numpy()
        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=False).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        edge_index = mod_adj.nonzero().t().contiguous()

        x, x1, x2 = features.clone().detach().cpu().numpy(), features.clone().detach().cpu().numpy(), \
            features.clone().detach().cpu().numpy()
        sampled_train_idx = np.random.choice(idx_train, int(idx_train.shape[0]), replace=False)
        for sti in sampled_train_idx:
            num = torch.nonzero(mod_adj[sti]).size()[0]
            sameclass_nodes = np.where(y[idx_train] == y[sti])[0]
            sameclass_id = np.random.choice(idx_train[sameclass_nodes], num, replace=True)
            x1[sti] = (x[sti] + np.mean(x[sameclass_id], axis=0)) / 2
            unclass_nodes = np.where(y[idx_train] != y[sti])[0]
            unclass_id = np.random.choice(idx_train[unclass_nodes], num, replace=True)
            x2[sti] = (x[sti] + np.mean(x[unclass_id], axis=0)) / 2
        x1, x2 = torch.Tensor(x1).to(self.device), torch.Tensor(x2).to(self.device)

        # training
        self.model.train()
        um, features_r, edge_index_r = self.um, features, edge_index
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            if epoch > self.epochs - self.finetune_epoch:
                output1, output2 = self.model(x1, edge_index, use_mp=True), self.model(x2, edge_index, use_mp=True)
                loss_cl = contrast_model(output1, output2)

                h = x2
                for i in range(self.model.num_layers - 1):
                    h = h @ self.model.fcs[i].weight.t()
                    h = gcn_conv(h, edge_index)
                    if self.model.ff_bias: h = h + self.model.fcs[i].bias
                    h = self.model.activation(h)
                    h = F.dropout(h, p=self.model.dropout, training=self.model.training)
                self.A_recon = h @ h.T
                self.A_recon = F.normalize(self.A_recon.detach().clone(), dim=1)
                loss_rec = torch.square(self.A_recon - mod_adj).sum(1).mean()

            # if epoch == self.epochs - 3:
            #     um, features_r, edge_index_r = self.A_aug(features, edge_index, idx_train, ratio=0.9)
            #     #um, features_r, edge_index_r = self.X_aug(features, edge_index, idx_train, ratio=1.0)
            # elif epoch == self.epochs - 20:
            #     um, features_r, edge_index_r = self.A_aug(features, edge_index, idx_train, ratio=0.7)
            #     #um, features_r, edge_index_r = self.X_aug(features, edge_index, idx_train, ratio=0.9)
            # elif epoch == self.epochs - 100:
            #     um, features_r, edge_index_r = self.A_aug(features, edge_index, idx_train, ratio=0.3)
            #     #um, features_r, edge_index_r = self.X_aug(features, edge_index, idx_train, ratio=0.8)
            # elif epoch == self.epochs - 150:
            #     um, features_r, edge_index_r = self.A_aug(features, edge_index, idx_train, ratio=0.1)
            #     #um, features_r, edge_index_r = self.X_aug(features, edge_index, idx_train, ratio=1.0)
            # else:
            #     um, features_r, edge_index_r = self.A_aug(features, edge_index, idx_train)
            #     #um, features_r, edge_index_r = self.X_aug(features, edge_index, idx_train)

            output = self.model(features_r, edge_index_r, use_mp=um)

            # output = self.model(features, edge_index, use_mp=False)
            output_log = F.log_softmax(output, dim=1)
            loss_train = F.nll_loss(output_log[idx_train], labels[idx_train])
            loss_train += 1. * loss_cl - 0.5 * loss_rec
            loss_train.backward()
            optimizer.step()
        acc_train = utils.accuracy(output[idx_train], labels[idx_train])

        return acc_train.item()

    def test(self, features, mod_adj, labels, idx_test):
        self.model.eval()
        edge_index = mod_adj.nonzero().t().contiguous()
        output = self.model(features, edge_index, True)
        output = F.log_softmax(output, dim=1)
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        return acc_test.item()


class MyNoisyGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, num_layers=2, dropout=0.5, num_node=0):
        super(MyNoisyGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.ff_bias = True  # Use bias for FF layers in default
        self.bns = nn.BatchNorm1d(hidden_channels, affine=False, track_running_stats=False)
        self.activation = F.relu
        self.noise_ratio_1 = 0.1

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(
            nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias))  # 1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias))  # 1
        self.reset_parameters()

    def reset_parameters(self):
        for mlp in self.fcs:
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def inject_noise(self, x, scale_noise):

        """
        Main function to sample noise
        ---
        Inputs:
            x: the embedding vector
            scale_noise (float,): The noise scale -- denoted as \beta in the paper

        Output:
            noise: The sampled produced noise to be added to the embedding vector
        """

        # Initiate a centred gaussian
        loc = torch.zeros(x.shape, dtype=torch.float32)
        scale = torch.ones(x.shape, dtype=torch.float32)
        noise = torch.distributions.Normal(loc, scale).sample()

        # Rescale the gaussian based on the noise ratio
        noise = scale_noise * noise

        return noise.to(x.device)

    def forward(self, x, edge_index, use_mp=True):
        for i in range(self.num_layers - 1):
            x = x @ self.fcs[i].weight.t()
            if use_mp: x = gcn_conv(x, edge_index)
            if self.ff_bias: x = x + self.fcs[i].bias
            if self.training:
                noise = self.inject_noise(x, self.noise_ratio_1)
                x = x + noise
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = x @ self.fcs[-1].weight.t()
        if use_mp: x = gcn_conv(x, edge_index)
        if self.ff_bias: x = x + self.fcs[-1].bias
        return x


class MyGCNJaccard(nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, num_layers=2, dropout=0.5, num_node=0):
        super(MyGCNJaccard, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.ff_bias = True  # Use bias for FF layers in default
        self.gcn_jaccard = GCNJaccard(nfeat=in_channels, nclass=out_channels,
                                      nhid=hidden_channels, dropout=dropout)

        self.bns = nn.BatchNorm1d(hidden_channels, affine=False, track_running_stats=False)
        self.activation = F.relu

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(
            nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias))  # 1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias))  # 1
        self.reset_parameters()

    def reset_parameters(self):
        for mlp in self.fcs:
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_mp=True):
        for i in range(self.num_layers - 1):
            x = x @ self.fcs[i].weight.t()
            if use_mp: x = gcn_conv(x, edge_index)
            if self.ff_bias: x = x + self.fcs[i].bias
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = x @ self.fcs[-1].weight.t()
        if use_mp: x = gcn_conv(x, edge_index)
        if self.ff_bias: x = x + self.fcs[-1].bias
        return x
