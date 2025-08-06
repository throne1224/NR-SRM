import numpy as np
import os
import csv
import networkx as nx
from networkx.algorithms.community.quality import modularity
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity

# 1. 土地利用集中度（Dominance Index）
def calculate_land_use_dominance(gdf, labels):
    gdf['cluster'] = labels
    dominance_dict = {}
    land_use_columns = ['utility_land', 'commercial_services', 'residential_land',
                        'industrial_land', 'green_square_land', 'road_transportation_land']
    for cluster in gdf['cluster'].unique():
        cluster_data = gdf[gdf['cluster'] == cluster]
        land_use_sums = cluster_data[land_use_columns].sum()
        total_land_use = land_use_sums.sum()
        if total_land_use == 0:
            dominance_dict[cluster] = np.nan
            continue
        land_use_proportions = land_use_sums / total_land_use
        dominance = land_use_proportions.max()
        dominance_dict[cluster] = dominance
    return np.nanmean(list(dominance_dict.values()))

# 2. 簇内中位数余弦相似度
def calculate_similarity(features, labels, n_clusters):
    if np.isnan(features).any():
        features = np.nan_to_num(features)
    cossim_mx = cosine_similarity(features)
    sim_dict = {}
    for c in range(n_clusters):
        ct_com = np.where(labels == c)[0]
        if len(ct_com) == 0:
            continue
        cossim_com = cossim_mx[ct_com[:, None], ct_com[None, :]]
        if len(ct_com) > 1:
            cossim = cossim_com[np.triu_indices(len(ct_com), k=1)]
        else:
            cossim = np.array([1.0])
        sim_dict[c] = np.mean(cossim)
    return np.median(list(sim_dict.values())) if sim_dict else np.nan

# 3. Calinski-Harabasz 指数
def calculate_calinski_harabasz_score(X, labels):
    return calinski_harabasz_score(X, labels)

# 4. 模块化指数
def calculate_modularity(adj_matrix, labels):
    G = nx.from_numpy_array(adj_matrix)
    communities = {}
    for node, label in enumerate(labels):
        if label not in communities:
            communities[label] = []
        communities[label].append(node)
    communities = list(communities.values())
    return modularity(G, communities)

# 5. 联合计数比（JCR）
def calculate_jcr(adj_matrix, labels):
    n_nodes = adj_matrix.shape[0]
    same_cluster = 0
    total_edges = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adj_matrix[i, j] == 1:
                total_edges += 1
                if labels[i] == labels[j]:
                    same_cluster += 1
    return same_cluster / total_edges if total_edges > 0 else 0.0

# 6. 综合指标
def calculate_composite_score(median_cosine, modularity, jcr):
    if np.isnan(median_cosine) or np.isnan(modularity) or np.isnan(jcr):
        return np.nan
    return round(median_cosine * modularity * jcr, 4)

def save_results(n_clusters, metric, linkage,
                 median_sim, ch_score, land_use_dominance,
                 modularity_score, jcr, composite_score):
    result_csv = 'cluster_result.csv'
    result_path = 'result'
    result_path_full = os.path.join(result_path, result_csv)

    if not os.path.exists(result_path_full):
        with open(result_path_full, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_head = [
                'n_clusters', 'linkage', 'metric',
                'Dominance Index', 'Calinski-Harabasz Score',
                'Median Cosine Similarity', 'Modularity', 'JCR', 'Composite Score'
            ]
            csv_writer.writerow(csv_head)

    with open(result_path_full, mode='a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_data = [
            n_clusters, linkage, metric,
            land_use_dominance, ch_score,
            median_sim, modularity_score, jcr, composite_score
        ]
        csv_writer.writerow(csv_data)