import numpy as np


def random_vector_gen(dimension: int, uniform: bool = False):
    """
    随机生成向量
    :param dimension: 维数
    :param uniform: 是否归一化，是：长度为1，否：取值范围0~1
    :return: 归一化后的随机向量
    """
    if uniform:
        vector = np.random.rand(dimension)
        length = np.linalg.norm(vector)
        return vector / length  # 归一化
    else:
        vector = np.random.rand(dimension)
        return vector
