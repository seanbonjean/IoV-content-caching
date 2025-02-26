from data import *
import v2i_entities
import utils


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
        RSU_entities[belong_RSU].serving_vehicles = []
        RSU_entities[belong_RSU].serving_vehicles.append(i)
        MBS_entities[belong_MBS].serving_vehicles = []
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
    if grand_tick_pass() == "end":
        break
    print()
    print("---Grand Time Slot: ", grand_time_slot)
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
        print("Small Time Slot: ", small_time_slot)
        for rsu in RSU_entities:
            memory_spent = np.inf
            while memory_spent > rsu_caching_memory[rsu.id]:
                rsu.agent.update_u()
                rsu.agent.sample_f_g(rsu.serving_vehicles)
                rsu.agent.update_zxq(W_matrix, RSU_entities, rsu.serving_vehicles)
                memory_spent = v2i_entities.constraint_memory(rsu.agent.x)
        print("RSU Loss (continuous x):", [rsu.agent.loss for rsu in RSU_entities])
        print("RSU total Loss (continuous x):", sum([rsu.agent.loss for rsu in RSU_entities]))
    # TODO MBS的决策算法
    # TODO 计算总的目标函数、审查全局约束条件（现在还不是总的，少MBS决策结果）
    print()
    # 各rsu内用户的delay
    local_loss = [v2i_entities.f_func(utils.discretization(rsu.agent.x), rsu.serving_vehicles) for rsu in RSU_entities]
    print("RSU Loss (discrete x):", local_loss)
    # 总delay
    global_loss = sum(local_loss)
    print("RSU total Loss (discrete x):", global_loss)
    # 各rsu是否满足内存约束，为布尔量列表
    memory_constraint = [v2i_entities.constraint_memory(rsu.agent.x) <= rsu.caching_memory for rsu in RSU_entities]
    # 内存约束是否全部满足
    memory_constraint_satisfied = all(memory_constraint)
    print("Memory Constraint Satisfied: ", memory_constraint_satisfied)
    # 各rsu是否满足cost约束，为布尔量列表
    local_cost_constraint = [v2i_entities.g_func(utils.discretization(rsu.agent.x), rsu.id) <= 0 for rsu in
                             RSU_entities]
    # 总cost
    global_cost_constraint = sum([v2i_entities.constraint_cost(rsu.agent.x) for rsu in RSU_entities])
    print("Global Cost Constraint: ", global_cost_constraint)
    # 总cost是否小于总cost约束
    global_cost_constraint_satisfied = global_cost_constraint <= sum(local_maximum_cache_cost)
    print("Global Cost Constraint Satisfied: ", global_cost_constraint_satisfied)
    # 所有约束是否全部满足
    constraint_satisfied = all((memory_constraint_satisfied, global_cost_constraint_satisfied))
    print("All Constraint Satisfied: ", constraint_satisfied)
