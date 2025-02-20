from data import *
import v2i_entities


def grand_tick_pass() -> str:
    """
    每个大时间片结束后，调用此函数以更新到下一时刻vehicle从属RSU等状态
    """
    global grand_time_slot
    grand_time_slot += 1
    if grand_time_slot >= GRAND_TIME_SLOT_NUM:
        return "end"
    for i, vehicle in enumerate(trajs):
        belong_MBS = vehicle['trajs'][grand_time_slot]['real_belongMBS']['index']
        belong_RSU = vehicle['trajs'][grand_time_slot]['real_belongRSU']['index']
        vehicle_entities[i].current_real_belong_MBS = belong_MBS
        vehicle_entities[i].current_real_belong_RSU = belong_RSU
        RSU_entities[belong_RSU].serving_vehicles.append(i)
        MBS_entities[belong_MBS].serving_vehicles.append(i)
        vehicle_entities[i].current_pred_belong_MBS = map(lambda x: x['index'],
                                                          vehicle['trajs'][grand_time_slot]['omegastep_belongMBS'])
        vehicle_entities[i].current_pred_belong_RSU = map(lambda x: x['index'],
                                                          vehicle['trajs'][grand_time_slot]['omegastep_belongRSU'])
    return "continue"


cloud_entity = v2i_entities.Cloud()
MBS_entities = [v2i_entities.MBS(mbs_caching_memory[i]) for i in range(MBS_NUM)]
RSU_entities = [v2i_entities.RSU(i, rsu_caching_memory[i]) for i in range(RSU_NUM)]
vehicle_entities = [v2i_entities.Vehicle() for _ in range(VEHICLE_NUM)]

grand_time_slot = -1  # 时间片计数器

# 大时间片的循环
while True:
    if grand_tick_pass() is "end":
        break
    # 大时间片开始前更新z
    # TODO 如何保持z中赋入的概率向量信息不被更新掉？
    for rsu in RSU_entities:
        new_z_dict = probability_table[grand_time_slot][rsu.id]
        for i in range(CONTENT_NUM):
            new_z_dict.setdefault(str(i), 0)  # 无概率content补0
        new_z = np.array([new_z_dict[str(i)] for i in range(CONTENT_NUM)])  # 字典类型转列表，生成numpy向量
        rsu.agent.reinit(new_z)
    # 小时间片，根据算法更新agent中的参数
    for small_time_slot in range(SMALL_TIME_SLOT_NUM):
        for rsu in RSU_entities:
            rsu.agent.update_u()
            rsu.agent.sample_f_g(rsu.serving_vehicles)
            rsu.agent.update_zxq(W_matrix, RSU_entities, rsu.serving_vehicles)

    # TODO MBS的决策算法
    # TODO 计算总的目标函数、审查全局约束条件
