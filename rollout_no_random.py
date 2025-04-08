import numpy as np
from itertools import combinations


class DynamicCacheOptimizer:
    def __init__(self, contents, sizes, capacity):
        """
        初始化缓存优化器
        :param contents: 内容ID列表
        :param sizes: 内容大小字典 {content_id: size}
        :param capacity: MBS总容量
        """
        self.contents = contents
        self.sizes = sizes
        self.capacity = capacity
        self.value_density = {}  # 动态计算的价值密度缓存

    def compute_value_density(self, predicted_requests, delay_gains):
        """
        计算每个内容的价值密度 (v_i/s_i = n_i * (b_i-a_i) / s_i)
        """
        self.value_density = {
            c: (predicted_requests.get(c, 0) * delay_gains.get(c, 0)) / self.sizes.get(c, 1)
            for c in self.contents
        }
        print(self.value_density)

    def generate_candidates(self, current_cache, top_k=5):
        """
        高效生成候选动作（带剪枝）
        :param current_cache: 当前缓存集合(set)
        :param top_k: 考虑价值密度最高的top_k个内容
        :return: 候选动作列表[set]
        """
        candidates = [set(current_cache)]  # 候选动作包含"保持现状"

        current_size = sum(self.sizes.get(c, 0) for c in current_cache)
        remaining_cap = self.capacity - current_size

        # 按价值密度降序排序内容（过滤无效内容）
        valid_contents = [c for c in self.contents if c in self.value_density]
        sorted_contents = sorted(
            valid_contents,
            key=lambda x: self.value_density.get(x, 0),
            reverse=True
        )[:top_k]  # 直接取top_k，避免多余排序

        # 1. 添加操作
        for content in sorted_contents:
            if content not in current_cache and self.sizes.get(content, 0) <= remaining_cap:
                new_cache = set(current_cache)
                new_cache.add(content)
                candidates.append(new_cache)

        # 2. 替换操作
        if current_cache:
            # replacing_contents = current_cache.copy()
            # replacing_size = current_size
            #
            # not_replaceable = False  # 指示当前replacing_contents中不再有可替换content
            # while not not_replaceable:
            #     for content in sorted_contents:  # 已移除top_k的切片，现在是完整的CONTENT_NUM个content按照value_density排序
            #         # 找到当前缓存中价值密度最低的内容
            #         worst_in_cache = min(
            #             replacing_contents,
            #             key=lambda x: self.value_density.get(x, float('inf'))
            #         )
            #         if content not in replacing_contents and sorted_contents.index(content) < sorted_contents.index(
            #                 worst_in_cache):
            #             # 检查替换后容量
            #             new_size = replacing_size - self.sizes.get(worst_in_cache, 0) + self.sizes.get(content, 0)
            #             if new_size <= self.capacity:
            #                 replacing_size = new_size
            #                 replacing_contents.remove(worst_in_cache)
            #                 replacing_contents.add(content)
            #                 break
            #     else:
            #         not_replaceable = True
            #
            # candidates.append(replacing_contents)

            # lower_k = 5
            # # 找到当前缓存中价值密度最低的内容
            # worst_contents = sorted(
            #     current_cache,
            #     key=lambda x: self.value_density.get(x, float('inf'))
            # )[:lower_k]
            #
            # for worst_in_cache in worst_contents:
            #     for content in sorted_contents:
            #         if content not in current_cache:
            #             # 检查替换后容量
            #             new_size = current_size - self.sizes.get(worst_in_cache, 0) + self.sizes.get(content, 0)
            #             if new_size <= self.capacity:
            #                 new_cache = set(current_cache)
            #                 new_cache.remove(worst_in_cache)
            #                 new_cache.add(content)
            #                 candidates.append(new_cache)

            # 找到当前缓存中价值密度最低的内容
            worst_in_cache = min(
                current_cache,
                key=lambda x: self.value_density.get(x, float('inf'))
            )

            for content in sorted_contents:
                if content not in current_cache:
                    # 检查替换后容量
                    new_size = current_size - self.sizes.get(worst_in_cache, 0) + self.sizes.get(content, 0)
                    if new_size <= self.capacity:
                        new_cache = set(current_cache)
                        new_cache.remove(worst_in_cache)
                        new_cache.add(content)
                        candidates.append(new_cache)

        return candidates

    def rollout_step(self, current_cache, predicted_requests, delay_gains):
        """
        执行单步Rollout优化
        :param current_cache: 当前缓存集合(set)
        :param predicted_requests: 预测请求次数 {content_id: requests}
        :param delay_gains: 各内容的延迟增益 {content_id: b_i-a_i}
        :return: 优化后的缓存集合(set)
        """
        try:
            # 计算价值密度
            self.compute_value_density(predicted_requests, delay_gains)

            # 生成候选动作
            candidates = self.generate_candidates(current_cache)

            best_total_gain = -np.inf
            best_cache = set(current_cache)

            for candidate in candidates:
                # 当前收益 = sum(n_i * (b_i-a_i))
                current_gain = sum(
                    predicted_requests.get(c, 0) * delay_gains.get(c, 0)
                    for c in candidate
                )

                # 模拟未来一步贪心策略
                future_gain = self.simulate_greedy_step(candidate, predicted_requests, delay_gains)

                # 选择总收益最大的动作
                if (current_gain + future_gain > best_total_gain) or \
                        (current_gain + future_gain == best_total_gain and len(candidate) > len(best_cache)):
                    best_total_gain = current_gain + future_gain
                    best_cache = candidate

            return best_cache

        except Exception as e:
            print(f"Rollout执行异常: {e}")
            return set(current_cache)  # 失败时返回原缓存

    def simulate_greedy_step(self, cache, predicted_requests, delay_gains):
        """
        模拟未来一步的贪心策略
        :return: 未来收益
        """
        try:
            remaining_cap = self.capacity - sum(self.sizes.get(c, 0) for c in cache)
            if remaining_cap <= 0:
                return 0

            # 按价值密度排序剩余内容
            remaining_contents = [
                c for c in self.contents
                if c not in cache and c in predicted_requests and c in delay_gains
            ]
            sorted_contents = sorted(
                remaining_contents,
                key=lambda x: (predicted_requests.get(x, 0) * delay_gains.get(x, 0)) / max(1, self.sizes.get(x, 1)),
                reverse=True
            )

            future_gain = 0
            for content in sorted_contents:
                content_size = self.sizes.get(content, 0)
                if content_size <= remaining_cap:
                    future_gain += predicted_requests.get(content, 0) * delay_gains.get(content, 0)
                    remaining_cap -= content_size
                    if remaining_cap <= 0:
                        break

            return future_gain

        except Exception as e:
            print(f"模拟贪心策略异常: {e}")
            return 0


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
    if not content_prob:
        # 计算每个内容的价值密度 = (请求次数 * 延迟增益) / 大小
        value_density = {
            c: (predicted_requests.get(c, 0) * delay_gains.get(c, 0)) / sizes.get(c, 1)
            for c in contents
        }
    else:
        # 计算每个内容的价值密度 = 内容进入该MBS的概率 * (请求次数 * 延迟增益) / 大小
        value_density = {
            c: content_prob.get(str(c), 0) * (predicted_requests.get(c, 0) * delay_gains.get(c, 0)) / sizes.get(c, 1)
            for c in contents
        }


    # 按价值密度降序排序内容
    sorted_contents = sorted(
        contents,
        key=lambda x: value_density.get(x, 0),
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


# ==================== 测试案例 ====================
if __name__ == "__main__":
    # 系统参数
    contents = ['A', 'B', 'C', 'D', 'E']
    sizes = {'A': 1, 'B': 3, 'C': 1, 'D': 2, 'E': 3}
    capacity = 4

    # 初始化优化器
    optimizer = DynamicCacheOptimizer(contents, sizes, capacity)

    # 测试案例1：基础场景（差异化延迟增益）
    # print("=== 测试案例1：差异化延迟增益 ===")
    predicted_requests = {'A': 1, 'B': 4, 'C': 2, 'D': 5, 'E': 3}
    delay_gains = {'A': 5, 'B': 10, 'C': 8, 'D': 12, 'E': 28}  # 各内容不同的(b_i-a_i)

    # current_cache = {'A'}
    # print(f"初始缓存: {current_cache}")
    # new_cache = optimizer.rollout_step(current_cache, predicted_requests, delay_gains)
    # print(f"优化后缓存: {new_cache} (预期应选择高延迟增益的D或B)")

    # 测试案例2：验证从最优状态出发
    print("\n=== 测试案例2：验证从最优状态出发 ===")
    # optimal_cache = {'B', 'D'}  # 手动计算的最优解
    # optimal_cache = {'D','C'}
    optimal_cache = compute_optimal_cache(contents, sizes, capacity, predicted_requests, delay_gains)
    print(f"当前最优缓存: {optimal_cache}")
    new_cache = optimizer.rollout_step(optimal_cache, predicted_requests, delay_gains)
    print(f"Rollout优化结果: {new_cache} ")

    # 测试案例3：新增高价值内容
    # print("\n=== 测试案例3：动态适应新增内容 ===")
    # predicted_requests['E'] = 6  # 新增内容E
    # sizes['E'] = 2
    # delay_gains['E'] = 15
    # optimizer.contents.append('E')
    #
    # print(f"新增内容E后当前缓存: {optimal_cache}")
    # new_cache = optimizer.rollout_step(optimal_cache, predicted_requests, delay_gains)
    # print(f"优化后缓存: {new_cache} (预期应替换为E+D或E+B)")
