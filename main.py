import data_read as data
import utils
import json

MBS_RADIUS = 600  # meters
RSU_RADIUS = 100  # meters

CENTER_LOCATION = (116.36032115, 39.911045075)

TRAJ_LEN = 7  # 拥有完整omega step的轨迹长度

# users = data.read_json("data/user/discarded_users.json")
# omega_steps = data.read_json("data/user/discarded_omega_steps.json")
users = data.read_json("data/user/all_filtered_users.json")
omega_steps = data.read_json("data/user/all_user_omega_step.json")
mbs_pos = data.read_mbs_rsu("data/mbs_rsu/mbs_xy.txt")
# rsu_pos = data.read_mbs_rsu("data/mbs_rsu/rsu_xy.txt") + data.read_mbs_rsu("data/mbs_rsu/rsu_added_xy_v8.txt")
rsu_pos = data.read_mbs_rsu("data/mbs_rsu/rsu_added_xy_v9.txt")

print()

# omega_steps = omega_steps[:18]

# 寻找每个点属于哪个RSU和MBS
for user in omega_steps:
    for point in user['trajs']:
        # 真实点
        # MBS
        MBS_distances = [utils.calculate_distance(point['real'], pos) for pos in mbs_pos]
        min_distance = min(MBS_distances)
        if min_distance > MBS_RADIUS:
            print(f"User {user['user_id']} "
                  f"trajectory point {utils.XYtoGPS(point['real'][0], point['real'][1], CENTER_LOCATION[0], CENTER_LOCATION[1])} "
                  f"is not in any MBS range")
            point['real_belongMBS'] = {'index': -1, 'loc': None, 'distance': min_distance}
        else:
            min_distance_index = MBS_distances.index(min_distance)
            point['real_belongMBS'] = {'index': min_distance_index, 'loc': mbs_pos[min_distance_index],
                                       'distance': min_distance}
        # RSU
        RSU_distances = [utils.calculate_distance(point['real'], pos) for pos in rsu_pos]
        min_distance = min(RSU_distances)
        if min_distance > RSU_RADIUS:
            print(f"User {user['user_id']} "
                  f"trajectory point {utils.XYtoGPS(point['real'][0], point['real'][1], CENTER_LOCATION[0], CENTER_LOCATION[1])} "
                  f"is not in any RSU range")
            point['real_belongRSU'] = {'index': -1, 'loc': None, 'distance': min_distance}
        else:
            min_distance_index = RSU_distances.index(min_distance)
            point['real_belongRSU'] = {'index': min_distance_index, 'loc': rsu_pos[min_distance_index],
                                       'distance': min_distance}
        # 预测点
        belongMBS = [{} for _ in range(len(point['omega_step']))]
        belongRSU = [{} for _ in range(len(point['omega_step']))]
        for i, omega_step in enumerate(point['omega_step']):
            # MBS
            MBS_distances = [utils.calculate_distance(omega_step, pos) for pos in mbs_pos]
            min_distance = min(MBS_distances)
            if min_distance > MBS_RADIUS:
                print(f"User {user['user_id']} "
                      f"omega step No. {i}: {utils.XYtoGPS(omega_step[0], omega_step[1], CENTER_LOCATION[0], CENTER_LOCATION[1])} "
                      f"is not in any MBS range")
                belongMBS[i] = {'index': -1, 'loc': None, 'distance': min_distance}
            else:
                min_distance_index = MBS_distances.index(min_distance)
                belongMBS[i] = {'index': min_distance_index, 'loc': mbs_pos[min_distance_index],
                                'distance': min_distance}
            # RSU
            RSU_distances = [utils.calculate_distance(omega_step, pos) for pos in rsu_pos]
            min_distance = min(RSU_distances)
            if min_distance > RSU_RADIUS:
                print(f"User {user['user_id']} "
                      f"omega step No. {i}: {utils.XYtoGPS(omega_step[0], omega_step[1], CENTER_LOCATION[0], CENTER_LOCATION[1])} "
                      f"is not in any RSU range")
                belongRSU[i] = {'index': -1, 'loc': None, 'distance': min_distance}
            else:
                min_distance_index = RSU_distances.index(min_distance)
                belongRSU[i] = {'index': min_distance_index, 'loc': rsu_pos[min_distance_index],
                                'distance': min_distance}
            point['omegastep_belongMBS'] = belongMBS
            point['omegastep_belongRSU'] = belongRSU

# 计算概率
for user in omega_steps:
    for point in user['trajs']:
        probability = {}
        for step_belongRSU in point['omegastep_belongRSU']:
            probability.setdefault(step_belongRSU['index'], 0)
            probability[step_belongRSU['index']] += 1
        step_length = len(point['omega_step'])
        for key, value in probability.items():
            probability[key] = value / step_length
        point['omega_RSU_probability'] = probability

print()

# 打印概率，统计准确率
total_count = 0
accurate_count = 0
for user in omega_steps:
    for point in user['trajs']:
        total_count += 1
        if point['real_belongRSU']['index'] == max(point['omega_RSU_probability'].items(), key=lambda x: x[1])[0]:
            if point['real_belongRSU']['index'] != -1:
                accurate_count += 1
        print(f"{user['user_id']};{point['real_belongRSU']['index']};{point['omega_RSU_probability']}")
        # print(f"User {user['user_id']} trajectory point {point['real']} belongs: "
        #       f"RSU No.{point['real_belongRSU']['index']} at position {point['real_belongRSU']['loc']}, "
        #       f"and the probability: {point['omega_RSU_probability']}")
print(f"Accuracy: {accurate_count / total_count}")

# # 单独打印预测不准确的点（经纬度）
# print("单独打印预测不准确的点（经纬度）：")
# for user in omega_steps:
#     for point in user['trajs']:
#         if point['real_belongRSU']['index'] != max(point['omega_RSU_probability'].items(), key=lambda x: x[1])[0]:
#             if point['real_belongRSU']['loc'] is not None:
#                 print(f"User {user['user_id']} trajectory point "
#                       f"{utils.XYtoGPS(point['real'][0], point['real'][1], CENTER_LOCATION[0], CENTER_LOCATION[1])}"
#                       " belongs: "
#                       f"RSU No.{point['real_belongRSU']['index']} at position "
#                       f"{utils.XYtoGPS(point['real_belongRSU']['loc'][0], point['real_belongRSU']['loc'][1],
#                                        CENTER_LOCATION[0], CENTER_LOCATION[1])}, "
#                       f"and the probability: {point['omega_RSU_probability']}")
#             else:
#                 print(f"User {user['user_id']} trajectory point "
#                       f"{utils.XYtoGPS(point['real'][0], point['real'][1], CENTER_LOCATION[0], CENTER_LOCATION[1])}"
#                       " belongs: no RSU, "
#                       f"and the probability: {point['omega_RSU_probability']}")

with open("data/result/results.json", 'w') as f:
    json.dump(omega_steps, f)

# 建立概率表
probability_table_user = [[{} for _ in range(len(rsu_pos))] for _ in range(TRAJ_LEN)]
for user in omega_steps:
    for time_slot, point in enumerate(user['trajs']):
        for rsu_num, probability in point['omega_RSU_probability'].items():
            probability_table_user[time_slot][rsu_num][user['user_id']] = probability

print()

user_content = [
    [0, 17, 18, 42, 63, 71, 84],
    [1, 11, 13, 21, 47, 58, 74],
    [6, 19, 24, 31, 34, 37, 40, 118, 119, ],
    [3, 10, 26, 99, 101, ],
    [5, 14, 16, 36, 38, 69, 73, 75, 78, 108, 109, ],
]


def get_user_related_content(user_id):
    related_content = -1
    for content_num, content_list in enumerate(user_content):
        if user_id in content_list:
            if related_content != -1:
                raise ValueError(f"用户{user_id}被分配到多个内容")
            related_content = content_num
    return related_content


probability_table_content = [[{} for _ in range(len(rsu_pos))] for _ in range(TRAJ_LEN)]
for time_slot, table in enumerate(probability_table_user):
    for rsu_id, rsu in enumerate(table):
        for user_id, probability in rsu.items():
            content_num = get_user_related_content(user_id)
            probability_table_content[time_slot][rsu_id].setdefault(content_num, 1)
            probability_table_content[time_slot][rsu_id][content_num] *= 1 - probability
for time_slot, table in enumerate(probability_table_content):
    for rsu_id, rsu in enumerate(table):
        for content_num, probability in rsu.items():
            probability_table_content[time_slot][rsu_id][content_num] = 1 - probability

print()

with open("data/result/table.json", 'w') as f:
    json.dump(probability_table_content, f)
