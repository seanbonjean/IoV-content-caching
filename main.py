import route_predict.data_read as data
import v2i_entities

# 准备数据

MBS_NUM = 20
RSU_NUM = 100
VEHICLE_NUM = 69

current_time_slot = -1  # 时间片计数器

user_content = [
    [0, 17, 18, 42, 63, 71, 84, 120, 122, 123, 126, 135, 136, 137, 138, 140, 144, 145, 146, 149, 155, 156, 159, ],
    [1, 11, 13, 21, 47, 58, 74, ],
    [6, 19, 24, 31, 34, 37, 40, 118, 119, 127, 128, ],
    [3, 10, 26, 99, 101, ],
    [5, 14, 16, 36, 38, 69, 73, 75, 78, 108, 109, 124, 130, 132, 134, 142, 143, 147, 148, 150, 153, 154, 158, ],
]

# # 各MBS所覆盖的RSU
# covering_RSU = [
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, ],
#     [0, 1, 2, 3, 4, 5, 6, 7, ],
#     [],
#     [],
# ]

cloud_entity = v2i_entities.Cloud()
MBS_entities = [v2i_entities.MBS() for _ in range(MBS_NUM)]
RSU_entities = [v2i_entities.RSU() for _ in range(RSU_NUM)]
vehicle_entities = [v2i_entities.Vehicle() for _ in range(VEHICLE_NUM)]

# !!! 注意数据中的user_id不连续，是因为之前有剔除，防止混淆；后面使用时换用0-69的连续id！
trajs = data.read_json("route_predict/data/result/results_final.json")
probability_table = data.read_json("route_predict/data/result/table_final.json")
pass

p_v2r = 1
G_v2r = 1
epsilon_v2r = 1
d_v2r = 1
sigma_v2r = 1

snr_v2r = p_v2r * G_v2r / (epsilon_v2r * d_v2r ** 2 * sigma_v2r ** 2)

p_v2m = 1
G_v2m = 1
epsilon_v2m = 1
d_v2m = 1
sigma_v2m = 1

snr_v2m = p_v2m * G_v2m / (epsilon_v2m * d_v2m ** 2 * sigma_v2m ** 2)

p_m2c = 1
G_m2c = 1
epsilon_m2c = 1
d_m2c = 1
sigma_m2c = 1

snr_m2c = p_m2c * G_m2c / (epsilon_m2c * d_m2c ** 2 * sigma_m2c ** 2)


def env_tick_pass():
    """
    每个时间片结束后，调用此函数以更新到下一时刻vehicle从属RSU等状态
    """
    global current_time_slot
    current_time_slot += 1
    for i, vehicle in enumerate(trajs):
        belong_MBS = vehicle['trajs'][current_time_slot]['real_belongMBS']
        belong_RSU = vehicle['trajs'][current_time_slot]['real_belongRSU']
        vehicle_entities[i].current_real_belong_MBS = belong_MBS
        vehicle_entities[i].current_real_belong_RSU = belong_RSU
        RSU_entities[belong_RSU].serving_vehicles.append(i)
        MBS_entities[belong_MBS].serving_vehicles.append(i)
        vehicle_entities[i].current_pred_belong_MBS = map(lambda x: x['index'],
                                                          vehicle['trajs'][current_time_slot]['omegastep_belongMBS'])
        vehicle_entities[i].current_pred_belong_RSU = map(lambda x: x['index'],
                                                          vehicle['trajs'][current_time_slot]['omegastep_belongRSU'])

# TODO 算法实现部分


# TODO 计算目标函数、审查约束条件
