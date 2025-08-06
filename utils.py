import numpy as np
import scipy.sparse as sp
import torch
import os
from sklearn.preprocessing import MinMaxScaler

EPS = 1e-15


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./", dataset="knowledge_graph", hops=48):
    print(f'Loading {dataset} dataset...')

    # Load feature matrix from knowledge graph
    features = np.loadtxt(os.path.join(path, 'feature_matrix_f1.csv'), delimiter=',', skiprows=1)
    features = features[:, 1:]  # Remove the first column (NODE_ID)
    features = normalize_features(features)  # Normalize features
    features = torch.FloatTensor(features)

    # Load adjacency matrix from knowledge graph
    adj = np.loadtxt(os.path.join(path, 'Spatial_matrix.csv'), delimiter=',')
    adj = sp.csr_matrix(adj)  # Ensure adj is in sparse matrix format
    adj = adj + sp.eye(adj.shape[0])  # Add self-loops
    adj = normalize_adj(adj)  # Normalize adjacency matrix
    adj = sparse_mx_to_torch_sparse_tensor(adj)  # Convert to torch sparse tensor

    # Load spatial distance matrix
    dis = np.loadtxt(os.path.join(path, 'Spatial_distance_matrix.csv'), delimiter=',')
    dis = normalize_distance_matrix(dis)  # Normalize distance matrix
    dis = torch.FloatTensor(dis)

    # Compute hops_m matrix
    hops_m = dis.numpy().copy()  # Start with the normalized distance matrix
    zero_entries = hops_m < hops  # Mark entries less than the hops threshold
    hops_m = 1 / (np.log(hops_m + 1e-15) + 1)  # Apply non-linear transformation
    hops_m[zero_entries] = 0  # Set entries less than hops to 0
    hops_m = torch.FloatTensor(hops_m)  # Convert to torch tensor

    return adj, features, dis, hops_m


def normalize_features(features):
    """Normalize feature matrix using Min-Max scaling."""
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    return features


def normalize_adj(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_distance_matrix(mx):
    """Normalize distance matrix using Min-Max scaling."""
    scaler = MinMaxScaler()
    mx = scaler.fit_transform(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def purge(dir, filename, best_epoch, spill_num):
    """Remove unnecessary checkpoint files."""
    del_list = [f'Epoch_{i}_{filename}' for i in range(0, best_epoch)]
    if spill_num > 0:
        tmp = [f'Epoch_{j}_{filename}' for j in range(best_epoch + 1, best_epoch + spill_num + 1)]
        del_list.extend(tmp)
    for f in os.listdir(dir):
        if f in del_list:
            os.remove(os.path.join(dir, f))