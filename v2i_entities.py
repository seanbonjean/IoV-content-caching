# Vehicle to Infrastructure Entities
import numpy as np
import projection
import utils
from data import *


def constraint_memory(x: np.array):
    """计算已缓存内容大小"""
    return sum([x[i] * content_size[i] for i in range(CONTENT_NUM)])


def constraint_cost(x: np.array):
    """计算单个rsu缓存cost"""
    return sum([x[i] * alpha[i] * content_size[i] for i in range(CONTENT_NUM)])


def f_func(x, user_list: list):
    """
    使用输入的x计算f(x)
    :param x: 输入值
    :param user_list: agent所在RSU服务的用户列表
    # :param vehicle_entities: （从main中）传入用户列表（保留未来mbs_id使用）
    :return: f(x)的值
    """
    f_value = 0.0
    for user_id in user_list:
        content_id = get_user_content(user_id)
        # mbs_id = vehicle_entities[user_id].current_real_belong_MBS  # 如果v2m的速率需要查表得到时启用
        f_value += x[content_id] * content_size[content_id] / v_v2r + \
                   (1 - x[content_id]) * content_size[content_id] / v_v2m
    return f_value


def g_func(x, rsu_id: int):
    """
    计算g(x)
    :param x:
    :param rsu_id:
    :return:
    """
    g_value = np.array([0.0 for _ in range(CONSTRAINT_NUM)])
    g_value[0] = constraint_cost(x) - local_maximum_cache_cost[rsu_id]
    return g_value


class Agent:
    def __init__(self, rsu_id: int):
        self.id = rsu_id
        # TODO 参数待确定
        self.alpha = 5.0
        self.beta = 5.0
        self.gamma = 5.0
        self.ksai = 0.5
        self.delta = 0.7 * (0.5 * self.ksai)

        memory_spent = np.inf
        while memory_spent > rsu_caching_memory[rsu_id]:  # 检查是否满足rsu memory限制，首次进入while为inf确保进入
            self.u = utils.random_vector_gen(CONTENT_NUM, uniform=True)  # 随机生成长度为1的向量，维数为content数量
            # 随机生成维数为content数量、各元素取值范围为0~1的向量（这和“长度为1”的含义不同），然后乘1-ksai
            self.z = (1 - self.ksai) * utils.random_vector_gen(CONTENT_NUM)
            self.x = self.z + self.delta * self.u
            memory_spent = constraint_memory(self.x)

        self.q_actual = np.array([0.0 for _ in range(CONSTRAINT_NUM)])
        self.q_eval = np.array([0.0 for _ in range(CONSTRAINT_NUM)])

        self.loss = 0.0  # 损失函数f_i,t
        self.constraint_func = np.array([0.0 for _ in range(CONSTRAINT_NUM)])  # 约束函数g_i,t
        """
        所有constraint应<=0
        constr: local的cost限制
        ---以下内容作废
        constr0: 缓存内容总大小限制
        constr1: local的cost限制
        constr2: x-1<=0
        constr3: -x<=0
        """

    def reinit(self, new_z: np.array):
        memory_spent = np.inf
        while memory_spent > rsu_caching_memory[self.id]:  # 检查是否满足rsu memory限制，首次进入while为inf确保进入
            self.u = utils.random_vector_gen(CONTENT_NUM, uniform=True)  # 随机生成长度为1的向量，维数为content数量
            # 随机生成维数为content数量、各元素取值范围为0~1的向量（这和“长度为1”的含义不同），然后乘1-ksai
            self.z = (1 - self.ksai) * new_z
            self.x = self.z + self.delta * self.u
            memory_spent = sum([self.x[i] * content_size[i] for i in range(CONTENT_NUM)])  # 计算已缓存内容大小

        self.q_actual = np.array([0.0 for _ in range(CONSTRAINT_NUM)])
        self.q_eval = np.array([0.0 for _ in range(CONSTRAINT_NUM)])

    def update_u(self):
        """
        Select vector u_i,t independently and uniformly at random
        """
        self.u = utils.random_vector_gen(CONTENT_NUM, uniform=True)

    def sample_f_g(self, user_list: list):
        """
        Sample f_i,t and g_i,t
        :param user_list: RSU负责服务的用户（从RSU对象属性中取值传入即可）
        """
        self.loss = f_func(self.x, user_list)
        self.constraint_func = g_func(self.x, self.id)

    def update_zxq(self, W_matrix: np.array, RSU_entities: list, user_list: list):
        """
        实现(9a)~(9d)
        :param W_matrix: mixing matrix
        :param RSU_entities: 传入所有RSU实例（用于读取各RSU的q值）
        :param user_list: RSU负责服务的用户（从RSU对象属性中取值传入即可）
        """
        self.q_eval = sum([W_matrix[self.id][j] * RSU_entities[j].agent.q_actual for j in range(RSU_NUM)])  # (9a)
        # (9b)
        g_gradient = CONTENT_NUM / self.delta * g_func(self.z + self.delta * self.u, self.id) * self.u
        g_gradient = g_gradient.reshape(CONTENT_NUM, CONSTRAINT_NUM)
        a = CONTENT_NUM / self.delta * f_func(self.z + self.delta * self.u, user_list) * self.u + \
            np.dot(g_gradient, self.q_eval.reshape(CONSTRAINT_NUM, 1)).reshape(CONTENT_NUM)
        self.z = projection.project_onto_box(self.z - self.alpha * a,
                                             np.array([0.0 for _ in range(CONTENT_NUM)]),
                                             np.array([1.0 - self.ksai for _ in range(CONTENT_NUM)]))
        self.x = self.z + self.delta * self.u  # (9c)
        # (9d)
        q_before_project = (1.0 - self.beta * self.gamma) * self.q_eval + self.gamma * self.constraint_func
        self.q_actual = projection.project_onto_box(q_before_project,
                                                    np.array([0.0 for _ in range(CONSTRAINT_NUM)]),
                                                    None)


class Cloud:
    def __init__(self):
        pass


class EdgeNode:
    def __init__(self, caching_memory: float):
        self.caching_memory = caching_memory  # 节点缓存大小
        self.cached_contents = []  # 节点上已缓存的content序号列表


class MBS(EdgeNode):
    def __init__(self, caching_memory: float):
        super().__init__(caching_memory)
        self.serving_vehicles = []  # 当前时刻正在服务（即正在与MBS通信）的vehicle序号列表
        # self.covering_RSU = covering_RSU  # 该MBS覆盖的RSU序号列表


class RSU(EdgeNode):
    def __init__(self, id: int, caching_memory: float):
        super().__init__(caching_memory)
        self.serving_vehicles = []  # 当前时刻正在服务（即正在与RSU通信）的vehicle序号列表
        self.id = id
        self.agent = Agent(id)


class Vehicle:
    def __init__(self):
        self.current_real_belong_MBS = -1
        self.current_real_belong_RSU = -1
        self.current_pred_belong_MBS = [-1, -1, -1, -1, -1]
        self.current_pred_belong_RSU = [-1, -1, -1, -1, -1]
