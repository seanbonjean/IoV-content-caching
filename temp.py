import utils
import data_read as data
import json
import matplotlib.pyplot as plt

CENTER_LOCATION = (116.36032115, 39.911045075)

# need_trans = [
#     [612.4243661125643, 477.3443340741371],
#     [528.2299515223295, -576.8052828656831],
#     [-167.6480869570531, 997.7717460608351],
#     [-168.40079576255957, 986.9793139026153],
#     [-169.1535066985484, 976.1868818190031],
# ]
#
# transformed = [utils.XYtoGPS(i[1], i[0], CENTER_LOCATION[0], CENTER_LOCATION[1]) for i in need_trans]
#
# print()
#
# point_A = [116.367502, 39.91533771428571]
#
# xy1 = utils.GPStoXY(point_A[0], point_A[1], CENTER_LOCATION[0], CENTER_LOCATION[1])
# gps1 = utils.XYtoGPS(xy1[0], xy1[1], CENTER_LOCATION[0], CENTER_LOCATION[1])
# xy2 = utils.GPStoXY(gps1[0], gps1[1], CENTER_LOCATION[0], CENTER_LOCATION[1])
# gps2 = utils.XYtoGPS(xy2[0], xy2[1], CENTER_LOCATION[0], CENTER_LOCATION[1])
#
# print()
#
# point_B = 116.367502, 39.91533771428571
#
# xy = utils.GPStoXY(point_B[0], point_B[1], CENTER_LOCATION[0], CENTER_LOCATION[1])
# gps = utils.XYtoGPS(xy[0], xy[1], CENTER_LOCATION[0], CENTER_LOCATION[1])
#
# print()
#
# # 用y,x尝试
#
# point_C = [116.367502, 39.91533771428571]
#
# yx = utils.GPStoXY(point_C[0], point_C[1], CENTER_LOCATION[0], CENTER_LOCATION[1])
# gps = utils.XYtoGPS(yx[1], yx[0], CENTER_LOCATION[0], CENTER_LOCATION[1])


# f = open("data/user/all_filtered_users.json", 'r')
# users = json.load(f)
# f.close()
#
# f = open("data/user/all_user_omega_step.json", 'r')
# omega_steps = json.load(f)
# f.close()

# for user in users:
#     print(user['user_id'])
# print("666")
# for user in omega_steps:
#     print(user['user_id'])

point1 = utils.GPStoXY(116.353649, 39.91671, CENTER_LOCATION[0], CENTER_LOCATION[1])
point2 = utils.GPStoXY(116.353872, 39.91747, CENTER_LOCATION[0], CENTER_LOCATION[1])

print(point1[0] - point2[0])
print(point1[1] - point2[1])
