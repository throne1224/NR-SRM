import os
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from evaluation import (
    calculate_similarity,
    calculate_calinski_harabasz_score,
    save_results,
    calculate_land_use_dominance,
    calculate_modularity,
    calculate_jcr,
    calculate_composite_score
)
from vision_w import run_visualizations

# 设置文件路径
result_path = 'result'
embedding_file = 'Epoch_252_lr_0.01_dropout_0.3_hidden1_32_hidden2_32_hidden3_16_output_11_patience_50_seed_42.csv'
adjacency_file = 'Spatial_matrix.csv'
feature_matrix_file = 'feature_matrix_f1.csv'
shapefile_path = r'D:\pythonProject\baseline_UR\ShangHai\shp\grid222geo.shp'

# 创建保存结果的路径
save_path = r"C:\Users\shu\Desktop\keshi\our"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 加载数据
def load_data():
    embedding_path = os.path.join(result_path, embedding_file)
    X = np.loadtxt(embedding_path, delimiter=' ')
    adj = np.loadtxt(adjacency_file, delimiter=',')
    features = np.loadtxt(feature_matrix_file, delimiter=',', skiprows=1)[:, 1:]
    return X, adj, features

# 运行层次聚类
def run_aggclustering(n_clusters, metric, linkage):
    X, adj, features = load_data()
    model = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters, connectivity=adj, metric=metric)
    model.fit(X)
    return X, adj, features, model.labels_

# 主函数
if __name__ == "__main__":
    n_clusters = 18
    metric = 'euclidean'
    linkage = 'ward'

    # 运行聚类
    X, adj_matrix, features, labels = run_aggclustering(n_clusters, metric, linkage)

    # 加载地理数据
    gdf = gpd.read_file(shapefile_path)
    if len(gdf) != len(labels):
        raise ValueError("聚类标签数量与地理数据不匹配")
    gdf['cluster'] = labels

    # 合并特征矩阵
    feature_matrix = pd.read_csv(feature_matrix_file)
    gdf = gdf.merge(feature_matrix, left_index=True, right_index=True, how='left')

    # 计算保留的6个指标
    land_use_dominance = calculate_land_use_dominance(gdf, labels)
    ch_score = calculate_calinski_harabasz_score(X, labels)
    median_sim = calculate_similarity(features, labels, n_clusters)
    modularity_score = calculate_modularity(adj_matrix, labels)
    jcr = calculate_jcr(adj_matrix, labels)
    composite_score = calculate_composite_score(median_sim, modularity_score, jcr)

    print(f"1. 土地利用集中度 (Dominance Index): {land_use_dominance:.3f}")
    print(f"2. Calinski-Harabasz 指数: {ch_score:.3f}")
    print(f"3. 簇内中位数余弦相似度 (Median Cosine Similarity): {median_sim:.3f}")
    print(f"4. 模块化指数 (Modularity): {modularity_score:.3f}")
    print(f"5. 联合计数比 (JCR): {jcr:.3f}")
    print(f"6. 综合指标 (Composite Score): {composite_score:.4f}")

    G = nx.Graph()
    for i in range(adj_matrix.shape[0]):
        for j in range(i + 1, adj_matrix.shape[1]):
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)


    run_visualizations(gdf, features[:, -7:], labels, G, save_path)

    # 保存所有指标
    save_results(
        n_clusters, metric, linkage,
        median_sim, ch_score, land_use_dominance,
        modularity_score, jcr, composite_score
    )
    print("\n所有指标已计算并保存到 cluster_result.csv")
