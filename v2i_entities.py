# Vehicle to Infrastructure Entities
import numpy as np
import utils
from data import *


class Agent:
    def __init__(self):
        # TODO 参数待确定
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0
        self.ksai = 0.0
        self.delta = 0.0

        self.u = utils.random_vector_gen(CONTENT_NUM, uniform=True)  # 随机生成长度为1的向量，维数为content数量
        # 随机生成维数为content数量、各元素取值范围为0~1的向量（这和“长度为1”的含义不同），然后乘1-ksai
        self.z = (1 - self.ksai) * utils.random_vector_gen(CONTENT_NUM)
        self.x = self.z + self.delta * self.u
        self.q_actual = np.array([0.0 for _ in range(CONSTRAINT_NUM)])
        self.q_pred = np.array([0.0 for _ in range(CONSTRAINT_NUM)])

        self.loss = 0.0  # 损失函数f_i,t
        self.constraint_func = np.array([0.0 for _ in range(CONSTRAINT_NUM)])  # 约束函数g_i,t
        """
        constr0: 缓存内容总大小限制
        constr1: local的cost限制
        constr2: x-1<=0
        constr3: -x<=0
        所有constraint应<=0
        """

    def update_u(self):
        """
        Select vector u_i,t independently and uniformly at random
        """
        self.u = utils.random_vector_gen(CONTENT_NUM, uniform=True)

    def sample_f_g(self, user_list: list, rsu_id: int):
        """
        Sample f_i,t and g_i,t
        :param user_list: RSU负责服务的用户（从RSU对象属性中取值传入即可）
        :param rsu_id: agent所在的RSU的序号
        # :param vehicle_entities: （从main中）传入用户列表（保留未来mbs_id使用）
        """
        for user_id in user_list:
            content_id = get_user_content(user_id)
            # mbs_id = vehicle_entities[user_id].current_real_belong_MBS  # 如果v2m的速率需要查表得到时启用
            self.loss += self.x[content_id] * content_size[content_id] / v_v2r + \
                         (1 - self.x[content_id]) * content_size[content_id] / v_v2m
            self.constraint_func[0] += sum(
                [self.x[i] * content_size[i] for i in range(CONTENT_NUM)]) - rsu_caching_memory[rsu_id]
            self.constraint_func[1] += sum([self.x[i] * alpha * content_size[i] for i in range(CONTENT_NUM)]) - \
                                       local_maximum_cache_cost[rsu_id]


def update_zxq(self):
    """
    实现(9a)~(9d)
    """
    pass


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
    def __init__(self, caching_memory: float):
        super().__init__(caching_memory)
        self.serving_vehicles = []  # 当前时刻正在服务（即正在与RSU通信）的vehicle序号列表
        self.agent = Agent()


class Vehicle:
    def __init__(self):
        self.current_real_belong_MBS = -1
        self.current_real_belong_RSU = -1
        self.current_pred_belong_MBS = [-1, -1, -1, -1, -1]
        self.current_pred_belong_RSU = [-1, -1, -1, -1, -1]
