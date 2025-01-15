import json


def read_json(path: str):
    f = open(path, 'r')
    data = json.load(f)
    f.close()
    return data


def read_mbs_rsu(path: str, src_sep: str = ',', x_pos=0, y_pos=1):
    mbs_or_rsu = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        properties = line.strip().split(src_sep)
        x, y = float(properties[x_pos]), float(properties[y_pos])
        mbs_or_rsu.append((x, y))
    return mbs_or_rsu
