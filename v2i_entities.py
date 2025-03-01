# Vehicle to Infrastructure Entities
import numpy as np
import projection
import utils
from data import *


def constraint_memory(x: np.ndarray):
    """计算已缓存内容大小"""
    return sum([x[i] * content_size[i] for i in range(CONTENT_NUM)])


def constraint_cost(x: np.ndarray):
    """计算单个rsu缓存cost"""
    return sum([x[i] * alpha[i] * content_size[i] for i in range(CONTENT_NUM)])


def f_func(x, user_list: list) -> float:
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


def delta_f(x, user_list: list, h=1e-5) -> np.ndarray:
    """
    计算f(x)在x处的导数
    :param x:
    :param user_list:
    :param h: 差分的步长，用来控制精度
    :return:
    """
    delta_x = np.empty_like(x)
    for i in range(CONTENT_NUM):
        x_ph = np.copy(x)
        x_mh = np.copy(x)
        x_ph[i] = x_ph[i] + h  # 计算f(x+h)和f(x-h)，这里的做法是为了保证原x不被修改，且各个i的x+h和x-h互不影响
        x_mh[i] = x_mh[i] - h
        delta_x[i] = (f_func(x_ph, user_list) - f_func(x_mh, user_list)) / (2 * h)

    return delta_x


def g_func(x, rsu_id: int, cons_num: int = -1) -> np.ndarray | int:
    """
    计算g(x)
    :param x:
    :param rsu_id:
    :param cons_num: 为-1时，计算所有constraint，否则计算对应constraint
    :return:
    """
    if cons_num == -1:
        # 计算所有constraint
        g_value = np.array([0.0 for _ in range(CONSTRAINT_NUM)])
        g_value[0] = constraint_cost(x) - local_maximum_cache_cost[rsu_id]
    elif cons_num == 0:
        # 计算第0个constraint
        g_value = constraint_cost(x) - local_maximum_cache_cost[rsu_id]
    else:
        raise ValueError("Invalid constraint number")
    return g_value


def delta_g(x, const_num: int, rsu_id: int, h=1e-5) -> np.ndarray:
    """
    计算g(x)在x处的导数
    :param x:
    :param const_num: 指定求第几个constraint的导数
    :param rsu_id:
    :param h:
    :return:
    """
    delta_x = np.empty_like(x)
    for i in range(CONTENT_NUM):
        x_ph = np.copy(x)
        x_mh = np.copy(x)
        x_ph[i] = x_ph[i] + h  # 计算f(x+h)和f(x-h)，这里的做法是为了保证原x不被修改，且各个i的x+h和x-h互不影响
        x_mh[i] = x_mh[i] - h
        delta_x[i] = (g_func(x_ph, rsu_id, cons_num=const_num) - g_func(x_mh, rsu_id, cons_num=const_num)) / (2 * h)

    return delta_x


class Agent:
    def __init__(self, rsu_id: int):
        self.id = rsu_id
        # TODO 参数待确定
        self.alpha = 0.3  # 预测概率向量占决策向量x的占比
        self.beta = 5.0
        self.delta = 0.01
        self.nita = 7.0

        self.x = np.array([0.0 for _ in range(CONTENT_NUM)])
        self.y = np.array([0.0 for _ in range(CONTENT_NUM)])
        self.lambd = np.array([0.0 for _ in range(CONSTRAINT_NUM)])

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

    def update_initial_x(self, prob_vect: np.ndarray) -> None:
        self.x = self.alpha * self.x + (1 - self.alpha) * prob_vect

    def update_x(self, A_matrix: np.ndarray, RSU_entities: list):
        p = sum([A_matrix[self.id][j] * RSU_entities[j].agent.y for j in range(RSU_NUM)])
        self.x = projection.project_onto_box(p,
                                             np.array([0.0 for _ in range(CONTENT_NUM)]),
                                             np.array([1.0 for _ in range(CONTENT_NUM)]))
    # def reinit(self, new_z: np.ndarray):
    #     memory_spent = np.inf
    #     while memory_spent > rsu_caching_memory[self.id]:  # 检查是否满足rsu memory限制，首次进入while为inf确保进入
    #         self.u = utils.random_vector_gen(CONTENT_NUM, uniform=True)  # 随机生成长度为1的向量，维数为content数量
    #         # 随机生成维数为content数量、各元素取值范围为0~1的向量（这和“长度为1”的含义不同），然后乘1-ksai
    #         self.z = (1 - self.ksai) * new_z
    #         self.x = self.z + self.delta * self.u
    #         memory_spent = sum([self.x[i] * content_size[i] for i in range(CONTENT_NUM)])  # 计算已缓存内容大小
    #
    #     self.q_actual = np.array([0.0 for _ in range(CONSTRAINT_NUM)])
    #     self.q_eval = np.array([0.0 for _ in range(CONSTRAINT_NUM)])

    def update_y(self, user_list: list):
        cons_sig = 0.0  # 步长右边那个sigma
        for i in range(CONSTRAINT_NUM):
            # i是约束函数的索引（第i个约束函数）
            if self.constraint_func[i] > 0:
                cons_sig += self.lambd[i] * delta_g(self.x, i, self.id)
        self.y = self.x - self.beta * (delta_f(self.x, user_list) + cons_sig)

    # def update_x(self, W_matrix: np.ndarray, RSU_entities: list):
    #     p = sum([W_matrix[self.id][j] * RSU_entities[j].agent.y for j in range(RSU_NUM)])
    #     self.x = projection.project_onto_box(p,
    #                                          np.array([0.0 for _ in range(CONTENT_NUM)]),
    #                                          np.array([1.0 for _ in range(CONTENT_NUM)]))
    #     # fine tuning
    #     memory_spent = np.inf
    #     while memory_spent > rsu_caching_memory[self.id]:  # 检查是否满足rsu memory限制，首次进入while为inf确保进入
    #         u = utils.random_vector_gen(CONTENT_NUM, uniform=True)
    #         self.x = (1 - self.ksai) * self.x + self.ksai * u
    #         self.x = projection.project_onto_box(self.x,
    #                                              np.array([0.0 for _ in range(CONTENT_NUM)]),
    #                                              np.array([1.0 for _ in range(CONTENT_NUM)]))
    #         memory_spent = constraint_memory(self.x)  # 计算已缓存内容大小

    def update_lambda(self):
        g_val = g_func(self.x, self.id)
        for i in range(CONSTRAINT_NUM):
            if g_val[i] < 0:
                self.lambd[i] = 0
            else:
                self.lambd[i] = g_val[i] / self.nita

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

    # def update_zxq(self, W_matrix: np.ndarray, RSU_entities: list, user_list: list):
    #     """
    #     实现(9a)~(9d)
    #     :param W_matrix: mixing matrix
    #     :param RSU_entities: 传入所有RSU实例（用于读取各RSU的q值）
    #     :param user_list: RSU负责服务的用户（从RSU对象属性中取值传入即可）
    #     """
    #     self.q_eval = sum([W_matrix[self.id][j] * RSU_entities[j].agent.q_actual for j in range(RSU_NUM)])  # (9a)
    #     # (9b)
    #     g_gradient = CONTENT_NUM / self.delta * g_func(self.z + self.delta * self.u, self.id) * self.u
    #     g_gradient = g_gradient.reshape(CONTENT_NUM, CONSTRAINT_NUM)
    #     a = CONTENT_NUM / self.delta * f_func(self.z + self.delta * self.u, user_list) * self.u + \
    #         np.dot(g_gradient, self.q_eval.reshape(CONSTRAINT_NUM, 1)).reshape(CONTENT_NUM)
    #     self.z = projection.project_onto_box(self.z - self.alpha * a,
    #                                          np.array([0.0 for _ in range(CONTENT_NUM)]),
    #                                          np.array([1.0 - self.ksai for _ in range(CONTENT_NUM)]))
    #     self.x = self.z + self.delta * self.u  # (9c)
    #     # (9d)
    #     q_before_project = (1.0 - self.beta * self.gamma) * self.q_eval + self.gamma * self.constraint_func
    #     self.q_actual = projection.project_onto_box(q_before_project,
    #                                                 np.array([0.0 for _ in range(CONSTRAINT_NUM)]),
    #                                                 None)


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
