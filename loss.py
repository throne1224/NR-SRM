import torch
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering


def clustering_loss(output, labels, n_clusters):
    """
    聚类损失函数
    """
    num_nodes = output.size(0)
    cluster_centers = torch.zeros(n_clusters, output.size(1)).to(output.device)
    for i in range(n_clusters):
        cluster_nodes = output[labels == i]
        cluster_center = torch.mean(cluster_nodes, dim=0, keepdim=True)
        cluster_centers[i] = cluster_center.squeeze(0)

    intra_cluster_loss = 0
    for i in range(n_clusters):
        cluster_nodes = output[labels == i]
        intra_cluster_loss += torch.mean(torch.sum((cluster_nodes - cluster_centers[i].unsqueeze(0)) ** 2, dim=1))

    inter_cluster_loss = torch.mean(torch.pow(torch.matmul(output, cluster_centers.T), 2))
    consistency_loss = torch.mean(torch.sum((output - cluster_centers[labels]) ** 2, dim=1))

    clustering_loss = intra_cluster_loss + inter_cluster_loss + consistency_loss
    return clustering_loss


def feature_similarity_loss(output, land_use_similarity):
    """
    特征一致性损失函数
    """
    embedding_similarity = F.cosine_similarity(output[:, None, :], output[None, :, :], dim=2)
    embedding_similarity = (embedding_similarity + 1) / 2  # Normalize to [0, 1]
    loss = F.mse_loss(embedding_similarity, land_use_similarity)
    return loss


def neighbors_loss(output, adj_dense, hops_m=None, temperature=0.1, ltype='div', hops=1):
    """
    邻接节点损失函数（结合对比学习和 hops_m 约束）
    """
    num_nodes = output.size(0)
    pdist = torch.norm(output[:, None] - output, dim=2, p=2)  # 计算节点嵌入之间的欧几里得距离
    inner_pro = torch.mm(output, output.T)  # 计算内积

    # 定义正样本和负样本的掩码
    pos_mask = adj_dense > 0
    neg_mask = adj_dense == 0

    # 计算正样本和负样本的损失
    pos_loss = torch.sum(pdist * pos_mask)
    neg_loss = torch.sum(pdist * neg_mask)

    # 如果提供了 hops_m 矩阵，则加入 hops 约束项
    if hops_m is not None and hops > 1:
        loss_hops = torch.sum(pdist * hops_m) + 1e-15  # hops 约束项：距离与 hops_m 的加权和
    else:
        loss_hops = 0

    if ltype == 'div':
        loss_train = pos_loss / (neg_loss + loss_hops + 1e-15)
    elif ltype == 'divreg':
        N_pos = torch.sum(pos_mask)
        N_neg = torch.sum(neg_mask)
        loss_train = (pos_loss * N_neg) / (N_pos * (neg_loss + loss_hops + 1e-15))
    else:
        raise ValueError("Unsupported ltype. Choose 'div' or 'divreg'.")

    return loss_train