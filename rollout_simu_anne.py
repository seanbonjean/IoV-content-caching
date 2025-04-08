import numpy as np

class DynamicCacheOptimizer:
    def __init__(self, contents, sizes, capacity):
        self.contents = contents
        self.sizes = np.array([sizes[c] for c in contents])
        self.capacity = capacity
        self.n_items = len(contents)

    def greedy_solution(self, values):
        value_density = values / self.sizes
        sorted_indices = np.argsort(-value_density)
        solution = np.zeros(self.n_items, dtype=int)
        remaining_capacity = self.capacity

        for idx in sorted_indices:
            if self.sizes[idx] <= remaining_capacity:
                solution[idx] = 1
                remaining_capacity -= self.sizes[idx]

        return solution

    def calculate_value(self, solution, values):
        return np.sum(solution * values)

    def calculate_weight(self, solution):
        return np.sum(solution * self.sizes)

    def simulated_annealing(self, initial_solution, values, initial_temp=100, cooling_rate=0.95, max_iter=50):
        current_solution = initial_solution.copy()
        current_value = self.calculate_value(current_solution, values)
        best_solution = current_solution.copy()
        best_value = current_value
        temp = initial_temp

        for _ in range(max_iter):
            neighbor_solution = current_solution.copy()
            flip_idx = np.random.randint(self.n_items)
            neighbor_solution[flip_idx] = 1 - neighbor_solution[flip_idx]

            if self.calculate_weight(neighbor_solution) > self.capacity:
                continue

            neighbor_value = self.calculate_value(neighbor_solution, values)
            delta = neighbor_value - current_value

            if delta > 0 or np.random.rand() < np.exp(delta / temp):
                current_solution = neighbor_solution.copy()
                current_value = neighbor_value

                if current_value > best_value:
                    best_solution = current_solution.copy()
                    best_value = current_value

            temp *= cooling_rate

        return best_solution

    def rollout(self, solution, values, max_steps=5):
        current_solution = solution.copy()

        for _ in range(max_steps):
            densities = values / self.sizes
            included_indices = np.where(current_solution == 1)[0]
            if len(included_indices) == 0:
                break

            lowest_density_idx = included_indices[np.argmin(densities[included_indices])]
            current_solution[lowest_density_idx] = 0

            remaining_capacity = self.capacity - self.calculate_weight(current_solution)

            best_idx = self.find_highest_density_item(current_solution, values, remaining_capacity)
            if best_idx != -1:
                current_solution[best_idx] = 1

        return current_solution

    def find_highest_density_item(self, solution, values, remaining_capacity):
        densities = values / self.sizes
        candidates = [(i, densities[i]) for i in range(self.n_items)
                      if solution[i] == 0 and self.sizes[i] <= remaining_capacity]
        if not candidates:
            return -1
        return max(candidates, key=lambda x: x[1])[0]

    def hybrid_optimization(self, predicted_requests, delay_gains, T=10, rollout_interval=3):
        values = np.array([predicted_requests[c] * delay_gains[c] for c in self.contents])

        initial_solution = self.greedy_solution(values)
        refined_solution = self.simulated_annealing(initial_solution, values)

        best_solution = refined_solution.copy()
        best_value = self.calculate_value(best_solution, values)

        for t in range(1, T + 1):
            if t % rollout_interval == 0:
                candidate_solution = self.rollout(best_solution, values)
                candidate_value = self.calculate_value(candidate_solution, values)

                if candidate_value > best_value:
                    best_solution = candidate_solution.copy()
                    best_value = candidate_value

        return best_solution, best_value

# ==================== 测试案例 ====================
if __name__ == "__main__":
    contents = ['A', 'B', 'C', 'D', 'E']
    sizes = {'A': 1, 'B': 3, 'C': 1, 'D': 2, 'E': 3}
    capacity = 4

    optimizer = DynamicCacheOptimizer(contents, sizes, capacity)

    predicted_requests = {'A': 1, 'B': 4, 'C': 2, 'D': 5, 'E': 3}
    delay_gains = {'A': 5, 'B': 10, 'C': 8, 'D': 12, 'E': 28}

    solution, total_value = optimizer.hybrid_optimization(predicted_requests, delay_gains)

    selected_contents = [contents[i] for i in range(len(contents)) if solution[i] == 1]

    print(f"Optimized Cached Contents: {selected_contents}")
    print(f"Total Value: {total_value}")
    print(f"Total Size Used: {optimizer.calculate_weight(solution)}")
