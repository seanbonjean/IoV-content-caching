import utils
import json
import matplotlib.pyplot as plt


def plot_singleuser_traj(user, show_pred: bool = False, slice_length: int = 11, x_range: tuple = None,
                         y_range: tuple = None):
    print("Now showing: user", user['user_id'])
    plt.figure(figsize=(10, 8))

    real_traj = user['real_traj']
    real_traj_GPS = [utils.XYtoGPS(i[0], i[1], CENTER_LOCATION[0], CENTER_LOCATION[1]) for i in real_traj]
    real_traj_GPS_lon = [i[0] for i in real_traj_GPS]
    real_traj_GPS_lat = [i[1] for i in real_traj_GPS]
    plt.plot(real_traj_GPS_lon, real_traj_GPS_lat, label="Real Trajectory", color="blue", linestyle='-', alpha=0.6)

    # 设置经纬度范围
    if x_range is not None:
        plt.xlim(x_range)
    if y_range is not None:
        plt.ylim(y_range)

    plt.title("User Trajectory")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    if show_pred:
        for i in range(slice_length):
            print(f"Now showing: user {user['user_id']}, predict slice {i}")
            plt.plot(real_traj_GPS_lon, real_traj_GPS_lat, label="Real Trajectory", color="blue", linestyle='-',
                     alpha=0.6)

            pred_traj = user['pred_traj'][i]['seqs']
            pred_traj_GPS = [utils.XYtoGPS(i[0], i[1], CENTER_LOCATION[0], CENTER_LOCATION[1]) for i in pred_traj]
            pred_traj_GPS_lon = [i[0] for i in pred_traj_GPS]
            pred_traj_GPS_lat = [i[1] for i in pred_traj_GPS]
            plt.scatter(pred_traj_GPS_lon, pred_traj_GPS_lat, color="red", label="(part of the) Predict Trajectory")

            if x_range is not None:
                plt.xlim(x_range)
            if y_range is not None:
                plt.ylim(y_range)

            plt.title("User Trajectory")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.legend()
            plt.grid(True)
            plt.show()


CENTER_LOCATION = (116.36032115, 39.911045075)
X_RANGE = (116.3518143, 116.368828)
Y_RANGE = (39.8996048, 39.92248535)

f = open("all_filtered_users.json", 'r')
# f = open("discarded_users.json", 'r')
users = json.load(f)
f.close()

f = open("all_user_omega_step.json", 'r')
# f = open("discarded_omega_steps.json", 'r')
omega_steps = json.load(f)
f.close()

plt.figure(figsize=(10, 8))
for user in users:
    real_traj = user['real_traj']
    real_traj_GPS = [utils.XYtoGPS(i[0], i[1], CENTER_LOCATION[0], CENTER_LOCATION[1]) for i in real_traj]
    real_traj_GPS_lon = [i[0] for i in real_traj_GPS]
    real_traj_GPS_lat = [i[1] for i in real_traj_GPS]
    plt.plot(real_traj_GPS_lon, real_traj_GPS_lat, label=f"user {user['user_id']}", color="blue", linestyle='-',
             alpha=0.6)
plt.xlim(X_RANGE)
plt.ylim(Y_RANGE)
plt.title("All User Trajectory")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()

for user in users:
    plot_singleuser_traj(user, show_pred=False, x_range=X_RANGE, y_range=Y_RANGE)
    # print("Press y to show prediction: ", end="")
    # if input() == 'y':
    #     plot_singleuser_traj(user, show_pred=True, x_range=X_RANGE, y_range=Y_RANGE)
