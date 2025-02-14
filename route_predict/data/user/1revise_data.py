import data_read as data
import json

"""
调换XY，弥补lstm里经纬度转换的代码错误
"""

users = data.read_json("filtered_users3.json")
omega_steps = data.read_json("user_omega_step3.json")

print()

for user in users:
    for i, point in enumerate(user['real_traj']):
        y, x = point[0], point[1]
        user['real_traj'][i] = (x, y)
    for slice in user['pred_traj']:
        for i, point in enumerate(slice['seqs']):
            y, x = point[0], point[1]
            slice['seqs'][i] = (x, y)

for user in omega_steps:
    for timestamp in user['trajs']:
        point = timestamp['real']
        y, x = point[0], point[1]
        timestamp['real'] = (x, y)
        for i, point in enumerate(timestamp['slice_lenth']):
            y, x = point[0], point[1]
            timestamp['slice_lenth'][i] = (x, y)

print()

f = open("filtered_users3_revised.json", 'w')
json.dump(users, f)
f.close()
f = open("user_omega_step3_revised.json", 'w')
json.dump(omega_steps, f)
f.close()
