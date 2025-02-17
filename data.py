import math
import route_predict.data_read as data_read
import v2i_entities

MBS_NUM = 20
RSU_NUM = 100
VEHICLE_NUM = 69
CONTENT_NUM = 5
CONSTRAINT_NUM = 4

user_content = [
    [0, 17, 18, 42, 63, 71, 84, 120, 122, 123, 126, 135, 136, 137, 138, 140, 144, 145, 146, 149, 155, 156, 159, ],
    [1, 11, 13, 21, 47, 58, 74, ],
    [6, 19, 24, 31, 34, 37, 40, 118, 119, 127, 128, ],
    [3, 10, 26, 99, 101, ],
    [5, 14, 16, 36, 38, 69, 73, 75, 78, 108, 109, 124, 130, 132, 134, 142, 143, 147, 148, 150, 153, 154, 158, ],
]
content_size = [1.0, 1.0, 1.0, 1.0, 1.0, ]  # TODO 待确定


def get_old_user_id(user_id_new: int) -> int:
    """
    进行新旧user id转换
    :param user_id_new: 新user id
    :return: 旧user id
    """
    return trajs[user_id_new]['user_id']


def get_user_content(user_id_new: int) -> list:
    """
    进行新旧user id转换并返回用户请求的对应内容序号
    :param user_id_new: 新user id
    :return: 新user id对应的该用户请求的内容序号
    """
    user_id = get_old_user_id(user_id_new)
    for i, users in enumerate(user_content):
        if user_id in users:
            return i
    raise ValueError(f"user_id {user_id} not found")


# # 各MBS所覆盖的RSU
# covering_RSU = [
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, ],
#     [0, 1, 2, 3, 4, 5, 6, 7, ],
#     [],
#     [],
# ]

# TODO 待确定
mbs_caching_memory = [7.0 for _ in range(MBS_NUM)]
rsu_caching_memory = [3.0 for _ in range(RSU_NUM)]
local_maximum_cache_cost = [10.0 for _ in range(RSU_NUM)]

# TODO 参数待确定
alpha = 0.7  # caching cost ratio
# Vehicle to RSU
p_v2r = 1
G_v2r = 1
epsilon_v2r = 1
d_v2r = 1  # TODO distance目前是一个定值，下同
sigma_v2r = 1
b_v2r = 1

# Vehicle to MBS
p_v2m = 1
G_v2m = 1
epsilon_v2m = 1
d_v2m = 1
sigma_v2m = 1
b_v2m = 1

# MBS to Cloud
p_m2c = 1
G_m2c = 1
epsilon_m2c = 1
d_m2c = 1
sigma_m2c = 1
b_m2c = 1

snr_v2r = p_v2r * G_v2r / (epsilon_v2r * d_v2r ** 2 * sigma_v2r ** 2)
snr_v2m = p_v2m * G_v2m / (epsilon_v2m * d_v2m ** 2 * sigma_v2m ** 2)
snr_m2c = p_m2c * G_m2c / (epsilon_m2c * d_m2c ** 2 * sigma_m2c ** 2)
v_v2r = b_v2r * math.log(1 + snr_v2r, 2)
v_v2m = b_v2m * math.log(1 + snr_v2m, 2)
v_m2c = b_m2c * math.log(1 + snr_m2c, 2)

# !!! 注意数据中的user_id不连续，是因为之前有剔除，防止混淆；后面使用时换用0-69的连续id！
trajs = data_read.read_json("route_predict/data/result/results_final.json")
probability_table = data_read.read_json("route_predict/data/result/table_final.json")
pass
