import numpy as np


def random_vector_gen(dimension: int, uniform: bool = False):
    """
    随机生成向量
    :param dimension: 维数
    :param uniform: 是否归一化，是：长度为1，否：取值范围0~1
    :return: 归一化后的随机向量
    """
    if uniform:
        vector = np.random.uniform(-1, 1, dimension)  # -1~1的均匀分布
        length = np.linalg.norm(vector)
        vector = vector / length  # 归一化
        return vector
    else:
        vector = np.random.rand(dimension)
        return vector


if __name__ == '__main__':
    vector = random_vector_gen(5)
    print(vector)
    print(type(vector))
