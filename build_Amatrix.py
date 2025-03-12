import numpy as np
import json
import pygmtools as pygm
import route_predict.data_read as data
from data import RSU_NUM

pygm.BACKEND = 'numpy'

TIME_SLOTS = 37

data = data.read_json("route_predict/data/result/results_0307.json")

A_matrix_list = []  # 每个时间片之间各有不同的A矩阵
for t in range(TIME_SLOTS - 1):
    A_matrix = [[0 for _ in range(RSU_NUM)] for _ in range(RSU_NUM)]  # A矩阵应为方阵，行列数皆为RSU_NUM
    for user in data:
        prev_rsu_num = user['trajs'][t]['real_belongRSU']['index']
        curr_rsu_num = user['trajs'][t + 1]['real_belongRSU']['index']
        A_matrix[curr_rsu_num][prev_rsu_num] += 1

    # # sinkhorn转化为doubly-stochastic
    # A_matrix = np.array(A_matrix)
    # A_matrix = pygm.sinkhorn(A_matrix)  # 转换为doubly-stochastic
    # A_matrix = A_matrix.tolist()  # 转换回list以便存储为json

    # # 取最大值，映射到0~1
    # max_num = 0
    # for i in range(RSU_NUM):
    #     for j in range(RSU_NUM):
    #         if A_matrix[i][j] > max_num:
    #             max_num = A_matrix[i][j]
    # for i in range(RSU_NUM):
    #     for j in range(RSU_NUM):
    #         A_matrix[i][j] = A_matrix[i][j] / max_num

    # 只按行归一化，全0行不归一化
    for i in range(RSU_NUM):
        sum_num = 0
        for j in range(RSU_NUM):
            sum_num += A_matrix[i][j]
        if sum_num == 0:
            continue
        for j in range(RSU_NUM):
            A_matrix[i][j] = A_matrix[i][j] / sum_num

    # # 只按列归一化，全0列不归一化
    # for j in range(RSU_NUM):
    #     sum_num = 0
    #     for i in range(RSU_NUM):
    #         sum_num += A_matrix[i][j]
    #     if sum_num == 0:
    #         continue
    #     for i in range(RSU_NUM):
    #         A_matrix[i][j] = A_matrix[i][j] / sum_num

    A_matrix_list.append(A_matrix)
pass

f = open("A_matrix.json", 'w')
json.dump(A_matrix_list, f)
