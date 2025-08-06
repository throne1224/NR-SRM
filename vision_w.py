import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.manifold import TSNE
import umap
import networkx as nx

# 定义自定义颜色映射
custom_colors = [
    "#FDDED7", "#F5BE8F", "#C1E0DB", "#CCD376", "#A28CC2",
    "#8498AB", "#4682B4", "#FCC41E", "#ABD7EC", "#81B21F",
    "#F59B7B", "#8FB4DC", "#FFDD8E", "#70CDBE", "#D9DEE7",
    "#CAC8EF", "#FF6347", "#482172"
]

# 可视化聚类结果的边界
def visualize_clusters_boundaries(gdf, labels, save_path=None, save_name='clusters_boundaries'):
    if len(gdf) != len(labels):
        raise ValueError("聚类标签数量与地理数据框行数不匹配。")
    gdf['cluster'] = labels

    # 确保颜色数量足够
    unique_clusters = len(np.unique(labels))
    if len(custom_colors) < unique_clusters:
        raise ValueError(f"提供的颜色数量不足，需要至少 {unique_clusters} 种颜色。")

    # 创建自定义颜色映射
    cmap = ListedColormap(custom_colors[:unique_clusters])

    fig, ax = plt.subplots(figsize=(10, 10))  # 修改这里
    gdf.plot(column='cluster', ax=ax, legend=True, cmap=cmap, edgecolor='white', linewidth=0.2)  # 修改这里

    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:len(np.unique(labels))], labels=labels[:len(np.unique(labels))])

    plt.box(False)  # 去掉坐标轴
    plt.show()

    # 如果指定了保存路径和名称，则保存图像
    if save_path and save_name:
        save_full_path = os.path.join(save_path, save_name + '.png')
        plt.savefig(save_full_path, bbox_inches='tight', dpi=300)
        print(f"边界图像已保存到：{save_full_path}")

def visualize_cluster_centers_with_boundaries(gdf, labels, custom_colors, save_path=None, save_name='cluster_centers_with_boundaries'):
    gdf['Longitude'] = pd.to_numeric(gdf['Longitude'], errors='coerce')
    gdf['Latitude'] = pd.to_numeric(gdf['Latitude'], errors='coerce')
    if gdf['Longitude'].hasnans or gdf['Latitude'].hasnans:
        print("警告：'Longitude' 或 'Latitude' 中存在非数值数据，已将其替换为 NaN。")
        gdf['Longitude'].fillna(gdf['Longitude'].mean(), inplace=True)
        gdf['Latitude'].fillna(gdf['Latitude'].mean(), inplace=True)
    gdf['cluster'] = labels
    cluster_centers = gdf.groupby('cluster')[['Longitude', 'Latitude']].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 10))  # 修改这里
    gdf.plot(column='cluster', ax=ax, legend=True, cmap=ListedColormap(custom_colors), edgecolor='white', linewidth=0.2)

    # 绘制聚类中心，使用红色标记
    ax.scatter(cluster_centers['Longitude'], cluster_centers['Latitude'], c='red', marker='x', s=100, label='Cluster Centers')

    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:len(np.unique(labels))+1], labels=labels[:len(np.unique(labels))+1])

    plt.box(False)  # 去掉坐标轴
    plt.show()

    # 如果指定了保存路径和名称，则保存图像
    if save_path and save_name:
        save_full_path = os.path.join(save_path, save_name + '.png')
        plt.savefig(save_full_path, bbox_inches='tight', dpi=300)
        print(f"边界图像已保存到：{save_full_path}")

# 调用所有可视化功能的接口函数
def run_visualizations(gdf, features, labels, G, save_path):
    visualize_clusters_boundaries(gdf, labels, save_path=save_path, save_name='clusters_w')
    visualize_cluster_centers_with_boundaries(gdf, labels, custom_colors, save_path=save_path, save_name='cluster_centers_w')