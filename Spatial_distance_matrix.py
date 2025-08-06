import geopandas as gpd
import numpy as np
from scipy.spatial import distance_matrix

# 读取 SHP 文件
gdf = gpd.read_file(r'D:\pythonProject\baseline_UR\ShangHai\shp\grid_222.shp')

# 确保数据为 WGS 1984 Web Mercator Auxiliary Sphere 投影
gdf = gdf.to_crs(epsg=3857)

# 提取网格的几何中心点坐标
coords = np.array([geom.centroid.coords[0] for geom in gdf.geometry])

# 初始化距离矩阵
num_grids = len(gdf)
dist_matrix = np.zeros((num_grids, num_grids))

# 使用欧氏距离计算
for i in range(num_grids):
    for j in range(i + 1, num_grids):
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 1000  # 转换为千米
        dist_matrix[i, j] = distance
        dist_matrix[j, i] = dist_matrix[i, j]  # 对称矩阵

# 保存为 CSV 文件
np.savetxt('Spatial_distance_matrix.csv', dist_matrix, delimiter=',', fmt='%.4f')