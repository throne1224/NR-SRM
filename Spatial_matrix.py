import pandas as pd
import numpy as np

# 读取Excel文件
file_path = 'adj.xls'  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 获取所有网格的唯一标识符
unique_ids = sorted(set(df['ID']).union(set(df['adj'])))

# 创建一个空的邻接矩阵
matrix_size = len(unique_ids)
adj_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

# 创建一个映射，将网格ID映射到矩阵的索引
id_to_index = {grid_id: idx for idx, grid_id in enumerate(unique_ids)}

# 遍历数据，填充邻接矩阵
for _, row in df.iterrows():
    id_index = id_to_index[row['ID']]
    adj_index = id_to_index[row['adj']]
    adj_matrix[id_index, adj_index] = 1
    adj_matrix[adj_index, id_index] = 1  # 如果是无向图，需要填充对称位置

# 将矩阵转换为DataFrame
adj_matrix_df = pd.DataFrame(adj_matrix)

# 输出结果
print("邻接矩阵：")
print(adj_matrix_df)

# 保存为CSV文件（不包含行和列标签）
adj_matrix_df.to_csv('adj_matrix.csv', index=False, header=False)
print("邻接矩阵已保存为adj_matrix.csv")