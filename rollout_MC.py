import numpy as np

# 模拟环境类
class MBSEnvironment:
    def __init__(self, contents, sizes, capacity, predicted_requests, delay_gains):
        self.contents = contents
        self.sizes = sizes
        self.capacity = capacity
        self.predicted_requests = predicted_requests
        self.delay_gains = delay_gains

    def transition(self, state, action):
        new_state = set(state)
        if action[0] == 'remove':
            new_state.discard(action[1])
        elif action[0] == 'add':
            if self._total_size(new_state | {action[1]}) <= self.capacity:
                new_state.add(action[1])
        elif action[0] == 'swap':
            new_state.discard(action[1])
            if self._total_size(new_state | {action[2]}) <= self.capacity:
                new_state.add(action[2])
        return frozenset(new_state)

    def reward(self, state):
        return sum(self.predicted_requests.get(c, 0) * self.delay_gains.get(c, 0) for c in state)

    def _total_size(self, cache_set):
        return sum(self.sizes.get(c, 0) for c in cache_set)

def greedy_policy(state, contents, sizes, predicted_requests, delay_gains, capacity):
    current_cache = set(state)
    used_capacity = sum(sizes[c] for c in current_cache)
    remaining = capacity - used_capacity
    remaining_items = [c for c in contents if c not in current_cache]
    densities = {
        c: (predicted_requests.get(c, 0) * delay_gains.get(c, 0)) / max(1, sizes.get(c, 1))
        for c in remaining_items
    }
    sorted_items = sorted(densities, key=lambda x: -densities[x])
    for c in sorted_items:
        if sizes[c] <= remaining:
            return ('add', c)
    return ('noop', None)

class MBSRolloutAgent:
    def __init__(self, env, rollout_policy, steps=10):
        self.env = env
        self.rollout_policy = rollout_policy
        self.steps = steps

    def _generate_actions(self, state):
        actions = []
        for c in self.env.contents:
            if c in state:
                actions.append(('remove', c))
            else:
                if self.env._total_size(state | {c}) <= self.env.capacity:
                    actions.append(('add', c))
        for c1 in state:
            for c2 in self.env.contents:
                if c2 not in state and c1 != c2:
                    new_state = (set(state) - {c1}) | {c2}
                    if self.env._total_size(new_state) <= self.env.capacity:
                        actions.append(('swap', c1, c2))
        return actions

    def simulate_rollout(self, s, a, depth=3):
        total_reward = 0
        s_next = self.env.transition(s, a)
        total_reward += self.env.reward(s_next)
        for _ in range(depth):
            a_next = self.rollout_policy(s_next)
            future_state = self.env.transition(s_next, a_next)
            if self.env._total_size(future_state) <= self.env.capacity:
                s_next = future_state
                total_reward += self.env.reward(s_next)
            else:
                break
        return total_reward

    def online_decision(self, state):
        candidates = self._generate_actions(state)
        results = [self.simulate_rollout(state, a) for a in candidates]
        best_action = candidates[np.argmax(results)]
        return best_action

# 测试
if __name__ == '__main__':
    contents = ['A', 'B', 'C', 'D', 'E']
    sizes = {'A': 1, 'B': 3, 'C': 1, 'D': 2, 'E': 3}
    capacity = 4
    predicted_requests = {'A': 1, 'B': 4, 'C': 2, 'D': 5, 'E': 3}
    delay_gains = {'A': 5, 'B': 10, 'C': 8, 'D': 12, 'E': 28}
    initial_cache = frozenset(['C', 'D'])

    env = MBSEnvironment(contents, sizes, capacity, predicted_requests, delay_gains)
    agent = MBSRolloutAgent(env, lambda s: greedy_policy(s, contents, sizes, predicted_requests, delay_gains, capacity))
    best_action = agent.online_decision(initial_cache)
    best_cache = env.transition(initial_cache, best_action)

    (best_action, set(best_cache), env._total_size(set(best_cache)))

    print(best_cache)