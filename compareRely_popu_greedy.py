def compute_optimal_cache(contents, sizes, capacity, predicted_requests, delay_gains, content_prob=None):
    """
    计算最优缓存集合（按价值密度从大到小选择，直到容量耗尽）

    :param contents: 内容ID列表
    :param sizes: 内容大小字典 {content_id: size}
    :param capacity: MBS总容量
    :param predicted_requests: 预测请求次数 {content_id: requests}
    :param delay_gains: 各内容的延迟增益 {content_id: b_i-a_i}
    :return: 最优缓存集合(set)
    """
    # 按价值密度降序排序内容
    sorted_contents = sorted(
        contents,
        key=lambda x: predicted_requests.get(x, 0),
        reverse=True
    )

    optimal_cache = set()
    remaining_capacity = capacity

    # 从高到低选择内容，直到容量耗尽
    for content in sorted_contents:
        content_size = sizes.get(content, 0)
        if content_size <= remaining_capacity:
            optimal_cache.add(content)
            remaining_capacity -= content_size
            if remaining_capacity <= 0:
                break

    return optimal_cache