import numpy as np
import json
import pygmtools as pygm
import route_predict.data_read as data
from data import RSU_NUM

pygm.BACKEND = 'numpy'

TIME_SLOTS = 7

data = data.read_json("route_predict/data/result/results_final.json")

A_matrix_list = []  # 每个时间片之间各有不同的A矩阵
for t in range(TIME_SLOTS - 1):
    A_matrix = [[0 for _ in range(RSU_NUM)] for _ in range(RSU_NUM)]  # A矩阵应为方阵，行列数皆为RSU_NUM
    for user in data:
        prev_rsu_num = user['trajs'][t]['real_belongRSU']['index']
        curr_rsu_num = user['trajs'][t + 1]['real_belongRSU']['index']
        A_matrix[curr_rsu_num][prev_rsu_num] += 1
    A_matrix = np.array(A_matrix)
    A_matrix = pygm.sinkhorn(A_matrix)  # 转换为doubly-stochastic
    A_matrix = A_matrix.tolist()  # 转换回list以便存储为json
    A_matrix_list.append(A_matrix)
pass

f = open("A_matrix.json", 'w')
json.dump(A_matrix_list, f)
