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

grand_time_slot = 0  # 时间片计数器

# 大时间片的循环
# 大时间片从1开始，舍弃0时间，因为0时间没有对应的A矩阵
while True:
    # 更新环境（用户位置等），并判断是否达到最后一个时间片需要结束循环
    if grand_tick_pass() == "end":
        break
    print()
    print("---Grand Time Slot: ", grand_time_slot)
    # for rsu in RSU_entities:
    #     rsu.agent.sample_f_g(rsu.serving_vehicles)
    #     rsu.agent.update_y(rsu.serving_vehicles)
    # # 这里要拆开的原因是，需要先更新所有rsu的y，然后才能互相communicate更新x
    # for rsu in RSU_entities:
    #     rsu.agent.update_x(W_matrix, RSU_entities)
    #     rsu.agent.update_lambda()
    for rsu in RSU_entities:
        # 构建概率向量
        prob_vect_dict = probability_table[grand_time_slot][rsu.id]
        for i in range(CONTENT_NUM):
            prob_vect_dict.setdefault(str(i), 0)  # 无概率content补0
        prob_vect = np.array([prob_vect_dict[str(i)] for i in range(CONTENT_NUM)])  # 字典类型转列表，生成numpy向量
        # 根据alpha比例，使用概率向量更新x向量的初始值
        rsu.agent.update_initial_x(prob_vect)
    # 小时间片
    for small_time_slot in range(SMALL_TIME_SLOT_NUM):
        print("Small Time Slot: ", small_time_slot)
        for rsu in RSU_entities:
            # 采样loss func: f 和 constraint func: g
            rsu.agent.sample_f_g(rsu.serving_vehicles)
            # 计算y值
            rsu.agent.update_y(rsu.serving_vehicles)
        # 这里要拆开的原因是，需要先更新所有rsu的y，然后才能互相communicate更新x

        # 这个feasible子集用于存储未被memory constraint终止fine-tuning的rsu集合
        RSU_entities_feasible = RSU_entities.copy()
        for rsu in RSU_entities:
            prev_x = rsu.agent.x  # 保存上一时刻的x，若下一时刻的x不满足约束条件，则回滚x
            # 更新p并投影到box，得到下一时刻t+1的决策x
            rsu.agent.update_x(A_matrix_list[grand_time_slot - 1], RSU_entities)
            # 根据t+1时的x更新lambda
            rsu.agent.update_lambda()
            # 离散化x
            decision_x = utils.discretization(rsu.agent.x)
            # 检查内存限制
            memory_spent = v2i_entities.constraint_memory(decision_x)
            if memory_spent > rsu_caching_memory[rsu.id]:
                rsu.agent.x = prev_x
                RSU_entities_feasible.remove(rsu)
        continuous_x = [rsu.agent.x for rsu in RSU_entities]
        print("RSU Decision x (continuous):", continuous_x)
        print("RSU Loss (continuous x):", [rsu.agent.loss for rsu in RSU_entities])
        print("RSU total Loss (continuous x):", sum([rsu.agent.loss for rsu in RSU_entities]))
    # TODO MBS的决策算法
    # TODO 计算总的目标函数、审查全局约束条件（现在还不是总的，少MBS决策结果）
    print()
    # 各rsu的决策x（连续形态）
    continuous_x = [rsu.agent.x for rsu in RSU_entities]
    print("RSU Decision x (continuous):", continuous_x)
    # 各rsu的决策x（离散形态）
    discrete_x = [utils.discretization(rsu.agent.x) for rsu in RSU_entities]
    print("RSU Decision x (discrete):", discrete_x)
    # 各rsu内用户的delay
    local_loss = [v2i_entities.f_func(utils.discretization(rsu.agent.x), rsu.serving_vehicles) for rsu in RSU_entities]
    print("RSU Loss (discrete x):", local_loss)
    # 总delay
    global_loss = sum(local_loss)
    print("RSU total Loss (discrete x):", global_loss)
    # 各rsu是否满足内存约束，为布尔量列表
    memory_constraint = [v2i_entities.constraint_memory(utils.discretization(rsu.agent.x)) <= rsu.caching_memory for rsu
                         in RSU_entities]
    # 内存约束是否全部满足
    memory_constraint_satisfied = all(memory_constraint)
    print("Memory Constraint Satisfied: ", memory_constraint_satisfied)
    # 各rsu是否满足cost约束，为布尔量列表
    local_cost_constraint = [v2i_entities.g_func(utils.discretization(rsu.agent.x), rsu.id) <= 0 for rsu in
                             RSU_entities]
    # 总cost
    global_cost_constraint = sum(
        [v2i_entities.constraint_cost(utils.discretization(rsu.agent.x)) for rsu in RSU_entities])
    print("Global Cost Constraint: ", global_cost_constraint)
    # 总cost是否小于总cost约束
    global_cost_constraint_satisfied = global_cost_constraint <= sum(local_maximum_cache_cost)
    print("Global Cost Constraint Satisfied: ", global_cost_constraint_satisfied)
    # 所有约束是否全部满足
    constraint_satisfied = all((memory_constraint_satisfied, global_cost_constraint_satisfied))
    print("All Constraint Satisfied: ", constraint_satisfied)
