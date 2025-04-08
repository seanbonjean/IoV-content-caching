import rollout_no_random as rollout

# 系统参数
# contents = ['A', 'B', 'C', 'D', 'E']
# sizes = {'A': 1, 'B': 3, 'C': 1, 'D': 2, 'E': 3}
contents = [0, 1, 2, 3, 4]
sizes = {0: 1, 1: 3, 2: 1, 3: 2, 4: 3}
capacity = 4

# 初始化优化器
optimizer = rollout.DynamicCacheOptimizer(contents, sizes, capacity)

# predicted_requests = {'A': 1, 'B': 4, 'C': 2, 'D': 5, 'E': 3}
# delay_gains = {'A': 5, 'B': 10, 'C': 8, 'D': 12, 'E': 28}  # 各内容不同的(b_i-a_i)
predicted_requests = {0: 1, 1: 4, 2: 2, 3: 5, 4: 3}
delay_gains = {0: 5, 1: 10, 2: 8, 3: 12, 4: 28}  # 各内容不同的(b_i-a_i)

optimal_cache = rollout.compute_optimal_cache(contents, sizes, capacity, predicted_requests, delay_gains)
print(f"当前最优缓存: {optimal_cache}")
new_cache = optimizer.rollout_step(optimal_cache, predicted_requests, delay_gains)
print(f"Rollout优化结果: {new_cache} ")
