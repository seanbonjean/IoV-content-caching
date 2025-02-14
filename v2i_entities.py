# Vehicle to Infrastructure Entities
import utils

CONTENT_NUM = 5


class Agent:
    def __init__(self):
        # TODO 参数待确定
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0
        self.ksai = 0.0
        self.delta = 0.0

        self.u = utils.random_vector_gen(CONTENT_NUM, uniform=True)  # 随机生成长度为1的向量，维数为content数量
        self.z = (1 - self.ksai) * utils.random_vector_gen(CONTENT_NUM)  # 随机生成维数为content数量、各元素取值范围为0~1的向量（这和“长度为1”的含义不同）



class Cloud:
    def __init__(self):
        pass


class EdgeNode:
    def __init__(self):
        self.cached_contents = []  # 节点上已缓存的content序号列表


class MBS(EdgeNode):
    def __init__(self):
        super().__init__()
        self.serving_vehicles = []  # 当前时刻正在服务（即正在与MBS通信）的vehicle序号列表
        # self.covering_RSU = covering_RSU  # 该MBS覆盖的RSU序号列表


class RSU(EdgeNode):
    def __init__(self):
        super().__init__()
        self.serving_vehicles = []  # 当前时刻正在服务（即正在与RSU通信）的vehicle序号列表
        self.agent = Agent()


class Vehicle:
    def __init__(self):
        self.current_real_belong_MBS = -1
        self.current_real_belong_RSU = -1
        self.current_pred_belong_MBS = [-1, -1, -1, -1, -1]
        self.current_pred_belong_RSU = [-1, -1, -1, -1, -1]
