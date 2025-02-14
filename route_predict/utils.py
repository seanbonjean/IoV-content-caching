import os
import math
import numpy as np

CONSTANTS_RADIUS_OF_EARTH = 6371000.  # meters (m)


def XYtoGPS(x, y, ref_lon, ref_lat):
    x_rad = float(x) / CONSTANTS_RADIUS_OF_EARTH
    y_rad = float(y) / CONSTANTS_RADIUS_OF_EARTH
    c = math.sqrt(x_rad * x_rad + y_rad * y_rad)

    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)

    ref_sin_lat = math.sin(ref_lat_rad)
    ref_cos_lat = math.cos(ref_lat_rad)

    if abs(c) > 0:
        sin_c = math.sin(c)
        cos_c = math.cos(c)

        lat_rad = math.asin(cos_c * ref_sin_lat + (x_rad * sin_c * ref_cos_lat) / c)
        lon_rad = (ref_lon_rad + math.atan2(y_rad * sin_c, c * ref_cos_lat * cos_c - x_rad * ref_sin_lat * sin_c))

        lat = math.degrees(lat_rad)
        lon = math.degrees(lon_rad)

    else:
        lat = math.degrees(ref_lat)
        lon = math.degrees(ref_lon)

    return lon, lat


def GPStoXY(lon, lat, ref_lon, ref_lat):
    # input GPS and Reference GPS in degrees
    # output XY in meters (m) X:North Y:East
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)

    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    ref_sin_lat = math.sin(ref_lat_rad)
    ref_cos_lat = math.cos(ref_lat_rad)

    cos_d_lon = math.cos(lon_rad - ref_lon_rad)

    arg = np.clip(ref_sin_lat * sin_lat + ref_cos_lat * cos_lat * cos_d_lon, -1.0, 1.0)
    c = math.acos(arg)

    k = 1.0
    if abs(c) > 0:
        k = (c / math.sin(c))

    x = float(k * (ref_cos_lat * sin_lat - ref_sin_lat * cos_lat * cos_d_lon) * CONSTANTS_RADIUS_OF_EARTH)
    y = float(k * cos_lat * math.sin(lon_rad - ref_lon_rad) * CONSTANTS_RADIUS_OF_EARTH)

    return x, y


def batch_GPStoXY(src_path, dst_path, ref_lon, ref_lat, lon_pos=2, lat_pos=3, src_sep=',', dst_sep=','):
    with open(src_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        properties = line.strip().split(src_sep)
        lon, lat = float(properties[lon_pos]), float(properties[lat_pos])
        x, y = GPStoXY(lon, lat, ref_lon, ref_lat)
        properties[lon_pos], properties[lat_pos] = str(x), str(y)
        lines[i] = dst_sep.join(properties) + '\n'

    with open(dst_path, 'w') as f:
        f.writelines(lines)


def batch_XYtoGPS(src_path, dst_path, ref_lon, ref_lat, x_pos=2, y_pos=3, src_sep=',', dst_sep=','):
    with open(src_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        properties = line.strip().split(src_sep)
        x, y = float(properties[x_pos]), float(properties[y_pos])
        lon, lat = XYtoGPS(x, y, ref_lon, ref_lat)
        properties[x_pos], properties[y_pos] = str(lon), str(lat)
        lines[i] = dst_sep.join(properties) + '\n'

    with open(dst_path, 'w') as f:
        f.writelines(lines)


def clear_dir(dir_path: str):
    if os.path.exists(dir_path):
        for file_name in os.listdir(dir_path):
            os.remove(os.path.join(dir_path, file_name))
        # print("Folder successfully removed: ", dir_path)
    else:
        raise Exception("No such path: ", dir_path)


def calculate_distance(point_1, point_2):
    return ((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) ** 0.5
