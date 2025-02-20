import numpy as np
from scipy.optimize import minimize


def project_onto_l2_ball(x, radius):
    norm_x = np.linalg.norm(x)
    if norm_x > radius:
        return (x / norm_x) * radius
    return x


def project_onto_box(x, lower_bound, upper_bound):
    return np.clip(x, lower_bound, upper_bound)




def project_onto_convex_set(x, constraint_function):
    """
    投影到任意凸集，constraint_function(x) <= 0 定义可行域
    """

    def objective(z):
        return np.linalg.norm(z - x) ** 2  # 目标是最小化 ||z - x||^2

    # 初始值
    z0 = x

    # 约束
    constraints = {'type': 'ineq', 'fun': constraint_function}

    # 运行优化
    res = minimize(objective, z0, constraints=constraints)

    return res.x if res.success else x  # 如果优化失败，返回原值


if __name__ == '__main__':
    # L2球示例
    z_prev = np.array([1.5, -0.5, 2.0])  # 之前的 z
    alpha = 0.1
    a_t = np.array([-1.0, 2.0, -0.5])  # 更新方向
    xi_t = 0.1
    X_radius = 1.0  # 这里假设 X_i 是单位球

    z_new = z_prev - alpha * a_t  # 先执行梯度下降
    z_projected = project_onto_l2_ball(z_new, (1 - xi_t) * X_radius)  # 进行投影
    print(z_projected)

    # box示例
    z_prev = np.array([1.5, -0.5, 2.0])
    alpha = 0.1
    a_t = np.array([-1.0, 2.0, -0.5])
    xi_t = 0.1
    X_lower = np.array([-1.0, -1.0, -1.0])
    X_upper = np.array([1.0, 1.0, 1.0])

    z_new = z_prev - alpha * a_t  # 梯度下降
    z_projected = project_onto_box(z_new, (1 - xi_t) * X_lower, (1 - xi_t) * X_upper)  # 投影
    print(z_projected)


    # 示例：投影到 L1 球上
    def l1_ball_constraint(x):
        return 1.0 - np.sum(np.abs(x))  # ||x||_1 <= 1


    z_prev = np.array([0.5, -0.5, 1.5])
    alpha = 0.1
    a_t = np.array([-0.5, 0.2, -0.8])

    z_new = z_prev - alpha * a_t
    z_projected = project_onto_convex_set(z_new, l1_ball_constraint)
    print(z_projected)
