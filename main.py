import data_read as data
import utils
import json

MBS_RADIUS = 600  # meters
RSU_RADIUS = 100  # meters

users = data.read_json("data/user/filtered_users_revised.json")
omega_steps = data.read_json("data/user/user_omega_step_revised.json")
mbs_pos = data.read_mbs_rsu("data/mbs_rsu/mbs_xy.txt")
rsu_pos = data.read_mbs_rsu("data/mbs_rsu/rsu_xy.txt") + data.read_mbs_rsu("data/mbs_rsu/rsu_added_xy.txt")

print()

for user in omega_steps:
    for point in user['trajs']:
        # 真实点
        # MBS
        MBS_distances = [utils.calculate_distance(point['real'], pos) for pos in mbs_pos]
        min_distance = min(MBS_distances)
        if min_distance > MBS_RADIUS:
            print(f"User {user['user_id']} trajectory point {point['real']} is not in any MBS range")
            point['real_belongMBS'] = {'index': -1, 'loc': None, 'distance': min_distance}
        else:
            min_distance_index = MBS_distances.index(min_distance)
            point['real_belongMBS'] = {'index': min_distance_index, 'loc': mbs_pos[min_distance_index],
                                       'distance': min_distance}
        # RSU
        RSU_distances = [utils.calculate_distance(point['real'], pos) for pos in rsu_pos]
        min_distance = min(RSU_distances)
        if min_distance > RSU_RADIUS:
            print(f"User {user['user_id']} trajectory point {point['real']} is not in any RSU range")
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
                print(f"User {user['user_id']} omega step No. {i}: {omega_step} is not in any MBS range")
                belongMBS[i] = {'index': -1, 'loc': None, 'distance': min_distance}
            else:
                min_distance_index = MBS_distances.index(min_distance)
                belongMBS[i] = {'index': min_distance_index, 'loc': mbs_pos[min_distance_index],
                                'distance': min_distance}
            # RSU
            RSU_distances = [utils.calculate_distance(omega_step, pos) for pos in rsu_pos]
            min_distance = min(RSU_distances)
            if min_distance > RSU_RADIUS:
                print(f"User {user['user_id']} omega step No. {i}: {omega_step} is not in any RSU range")
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

with open("data/result/results.json", 'w') as f:
    json.dump(omega_steps, f)
