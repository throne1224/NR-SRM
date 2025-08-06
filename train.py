from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering
from utils import load_data, purge
from models import NodeAggregationModel
import csv
import os

# 导入损失函数模块
import loss


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=50, help='Early stopping control.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--input_feat_dim', type=int, default=11, help='Number of input features.')
parser.add_argument('--hidden_dim1', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--hidden_dim2', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--hidden_dim3', type=int, default=16, help='Number of hidden units.')  # 新增隐藏层维度
parser.add_argument('--output', type=int, default=11, help='Output dim.')
parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads.')  # 新增多头注意力头数
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.1, help='Negative slope for LeakyReLU activation.')
parser.add_argument('--lambda_cluster', type=float, default=0.2, help='Weight for clustering loss.')
parser.add_argument('--lambda_nei', type=float, default=0.2, help='Weight for functional similarity loss.')
parser.add_argument('--lambda_func', type=float, default=0.6, help='Weight for economic similarity loss.')
parser.add_argument('--n_clusters', type=int, default=14, help='Number of clusters')
parser.add_argument('--hops', type=int, default=48, help='Hops parameter for hops_m matrix.')
parser.add_argument('--ltype', type=str, default='div', choices=['div', 'divreg'], help='Type of loss function.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, dis, hops_m = load_data()
print("Data loaded successfully.")

# Normalize features to [0, 1] using MinMaxScaler
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features.cpu().numpy())
features = torch.tensor(features_normalized, dtype=torch.float32)
if args.cuda:
    features = features.cuda()
    adj = adj.cuda()
    dis = dis.cuda()
    hops_m = hops_m.cuda()

# Convert sparse adjacency matrix to dense (if necessary)
adj_dense = adj.to_dense() if adj.is_sparse else adj

# Model and optimizer
model = NodeAggregationModel(input_feat_dim=args.input_feat_dim,
                             hidden_dim1=args.hidden_dim1,
                             hidden_dim2=args.hidden_dim2,
                             hidden_dim3=args.hidden_dim3,
                             num_heads=args.num_heads,
                             dropout=args.dropout,
                             alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = StepLR(optimizer, step_size=200, gamma=0.1)

if args.cuda:
    model.cuda()

# Calculate similarity matrices
land_use_features = features[:, 4:11]
land_use_similarity = F.cosine_similarity(land_use_features[:, None, :], land_use_features[None, :, :], dim=2)
land_use_similarity = (land_use_similarity + 1) / 2

# Initialize best_loss and best_epoch
best_loss = float('inf')
best_epoch = -1
result_path = './result/'
if not os.path.exists(result_path):
    os.mkdir(result_path)

# Train model
t_total = time.time()
loss_list = []
for epoch in range(args.epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, output_sfx = model(features, adj_dense)  # Use adj_dense if adj is sparse

    # Calculate losses using functions from loss.py
    loss_neighbors_term = loss.neighbors_loss(output, adj_dense, hops_m=hops_m, ltype=args.ltype, hops=args.hops)
    spectral = SpectralClustering(n_clusters=args.n_clusters, affinity='nearest_neighbors', random_state=args.seed)
    cluster_labels = spectral.fit_predict(output.detach().cpu().numpy())
    loss_cluster_term = loss.clustering_loss(output, torch.tensor(cluster_labels).to(output.device), args.n_clusters)
    loss_feature_term = loss.feature_similarity_loss(output, land_use_similarity)

    # Total loss
    loss_train = (args.lambda_nei * loss_neighbors_term +
                  args.lambda_cluster * loss_cluster_term +
                  args.lambda_func * loss_feature_term)

    loss_train.backward()
    optimizer.step()

    loss_list.append(loss_train.item())

    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.5f}'.format(loss_train.item()),
          'loss_neighbors: {:.5f}'.format(loss_neighbors_term.item()),
          'loss_cluster: {:.5f}'.format(loss_cluster_term.item()),
          'loss_feature: {:.5f}'.format(loss_feature_term.item()),
          'time: {:.4f}s'.format(time.time() - t))

    if loss_train < best_loss:
        best_loss = loss_train.item()
        best_epoch = epoch
        save_path = os.path.join(result_path, 'best_model.pth')
        torch.save(model.state_dict(), save_path)

    if epoch >= 200:
        save_name = 'lr_{}_dropout_{}_hidden1_{}_hidden2_{}_hidden3_{}_output_{}_patience_{}_seed_{}.csv'.format(
            args.lr, args.dropout, args.hidden_dim1, args.hidden_dim2, args.hidden_dim3, args.output, args.patience, args.seed)
        intermediate_file_path = os.path.join(result_path, 'Epoch_{}_'.format(epoch) + save_name)
        np.savetxt(intermediate_file_path, output.detach().cpu().numpy())

        if epoch > 200 + args.patience and loss_train > np.average(loss_list[-args.patience:]):
            best_epoch = loss_list.index(min(loss_list))
            print('Lose patience, stop training...')
            print('Best epoch: {}'.format(best_epoch))
            purge(result_path, save_name, best_epoch, epoch - best_epoch)
            break

        if epoch == args.epochs - 1:
            print('Last epoch, saving...')
            best_epoch = epoch
            purge(result_path, save_name, best_epoch, 0)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Save training results to CSV
result_csv = 'result.csv'
result_file_path = os.path.join(result_path, result_csv)
if not os.path.exists(result_file_path):
    with open(result_file_path, 'w', newline='') as f:
        csv_write = csv.writer(f)
        csv_head = ['epoch', 'loss', 'output', 'hidden1', 'hidden2', 'hidden3', 'lr', 'dropout', 'patience', 'seed']
        csv_write.writerow(csv_head)

with open(result_file_path, 'a', newline='') as f:
    csv_write = csv.writer(f)
    csv_write.writerow([best_epoch, min(loss_list), args.output, args.hidden_dim1, args.hidden_dim2, args.hidden_dim3, args.lr, args.dropout, args.patience, args.seed])