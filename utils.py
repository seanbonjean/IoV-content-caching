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


def discretization(x: np.ndarray) -> np.array:
    # 使用阈值将值转换为 0 或 1
    threshold = 0.5
    discrete_array = (x >= threshold).astype(int)
    return discrete_array


def knapsack(weights, values, capacity, scale=10):
    """
    0/1 背包问题的动态规划解法，支持浮点型的 capacity 通过放大转换为整数。
    :param weights: 物品重量列表（可以是浮点数）
    :param values: 物品价值列表
    :param capacity: 背包容量（可以是浮点数）
    :param scale: 放大比例，默认10倍（即保留1位小数）
    :return: 最大总价值和选中的物品索引列表
    """
    # 放大 capacity 和 weights，转换为整数
    capacity = int(round(capacity * scale))
    weights = [int(round(w * scale)) for w in weights]

    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] > w:
                dp[i][w] = dp[i - 1][w]
            else:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])

    # 回溯找出选中的物品
    selected = []
    i, w = n, capacity
    while i > 0 and w > 0:
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)  # 选中当前物品
            w -= weights[i - 1]
            i -= 1
        else:
            i -= 1  # 未选中当前物品，继续检查前一个物品

    selected.reverse()  # 反转列表使物品按原始顺序排列
    return dp[n][capacity], selected



if __name__ == '__main__':
    # vector = random_vector_gen(5)
    # print(vector)
    # print(type(vector))

    # print(discretization(np.array([0.1, 0.2, 0.3, 0.4, 0.5])))
    # print(type(discretization(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))))

    weights = [2, 3, 7, 7]
    values = [10, 20, 40, 50]
    capacity = 6
    print(knapsack(weights, values, capacity))
