import numpy as np
import json
import route_predict.data_read as data_read
from route_predict.utils import calculate_distance
from data import RSU_NUM

DISTANCE_THRESHOLD = 300

rsu_pos = data_read.read_mbs_rsu("route_predict/data/mbs_rsu/rsu_xy_final.txt")

# 计算RSU间距离
distance_matrix = []
for self_pos in rsu_pos:
    distance_matrix.append([calculate_distance(self_pos, target_pos) for target_pos in rsu_pos])

# 构建01矩阵
W_matrix = []
for row in distance_matrix:
    W_matrix_row = []
    for distance in row:
        if distance <= DISTANCE_THRESHOLD:
            W_matrix_row.append(1)
        else:
            W_matrix_row.append(0)
    W_matrix.append(W_matrix_row)

# 使W矩阵满足doubly-stochastic
for i, row in enumerate(W_matrix):
    # 论文方法：1/n 和 1-sum(1/n)
    new_row = list(map(lambda x: x / RSU_NUM, row))  # 全部除以n
    total = sum(new_row[:i] + new_row[i + 1:])  # 去除i=j的数据后的和
    new_row[i] = 1 - total
    W_matrix[i] = new_row

json.dump(W_matrix, open("W_matrix.json", 'w'))
W_matrix = np.array(W_matrix)
pass
