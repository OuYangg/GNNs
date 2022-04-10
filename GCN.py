
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 1 11:23:51 2022
@author: Arvin Ou
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.sparse as sp
import argparse
# Two layer gcn
class GCNLayer(nn.Module):
    def __init__(self,input_features,output_features,bias=False):
        super(GCNLayer,self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weights = nn.Parameter(torch.FloatTensor(input_features,output_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1./math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std,std)
        if self.bias is not None:
            self.bias.data.uniform_(-std,std)

    def forward(self,adj,x):
        support = torch.mm(x,self.weights)
        output = torch.spmm(adj,support)
        if self.bias is not None:
            return output+self.bias
        return output

class GCN(nn.Module):
    def __init__(self,input_size,hidden_size,num_class,dropout,bias=False):
        super(GCN,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_class = num_class
        self.gcn1 = GCNLayer(input_size,hidden_size,bias=bias)
        self.gcn2 = GCNLayer(hidden_size,num_class,bias=bias)
        self.dropout = dropout
    def forward(self,adj,x):
        x = F.relu(self.gcn1(adj,x))
        x = F.dropout(x,self.dropout,training=self.training)
        x = self.gcn2(adj,x)
        return F.log_softmax(x,dim=1)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def prepare_data(features,edge_list,labels):
    """准备输入数据
    Params:
        features:网络节点总数ee
        edge_list:连边列表e
    """
    # 构造输入特征
    features = sp.csr_matrix(features,dtype=np.float64)
    adj = sp.coo_matrix((np.ones(edge_list.shape[0]), (edge_list[:, 0], edge_list[:, 1])), shape=(),
                        dtype=np.int64)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    labels = torch.LongTensor(labels)

    return features,adj,labels

def load_data(path="./cora/", dataset="cora"):
    """读取引文网络数据cora"""
    print('Loading {} dataset...'.format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str)) # 使用numpy读取.txt文件
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32) # 获取特征矩阵
    labels = encode_onehot(idx_features_labels[:, -1]) # 获取标签

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train_gcn(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(adj,features)
    loss = F.nll_loss(output[idx_train],labels[idx_train])
    acc = accuracy(output[idx_train],labels[idx_train])
    loss.backward()
    optimizer.step()
    loss_val = F.nll_loss(output[idx_val],labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss.item()),
          'acc_train: {:.4f}'.format(acc.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(adj,features)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    args = parser.parse_args()
    np.random.seed(args.seed)
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    model = GCN(features.shape[1],args.hidden,labels.max().item() + 1,dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        train_gcn(epoch)
