import matplotlib.pyplot as plt
import time
from data import *
import v2i_entities
import utils
import rollout_no_random as rollout

# import rollout_no_random as rollout2

log_recorder = open("log_" + time.strftime("%m%d_%H_%M_%S", time.localtime()) + ".txt", "w")
pred_log = open("pred.txt", "w")
real_log = open("real.txt", "w")


def grand_tick_pass() -> str:
    """
    每个大时间片结束后，调用此函数以更新到下一时刻vehicle从属RSU等状态
    """
    global grand_time_slot
    grand_time_slot += 1
    if grand_time_slot >= GRAND_TIME_SLOT_NUM:
        return "end"
    for rsu in RSU_entities:
        rsu.serving_vehicles = []
        rsu.pred_serving_vehicles = []
    for mbs in MBS_entities:
        mbs.serving_vehicles = []
    for i, vehicle in enumerate(trajs):
        real_belong_MBS = vehicle['trajs'][grand_time_slot]['real_belongMBS']['index']
        real_belong_RSU = vehicle['trajs'][grand_time_slot]['real_belongRSU']['index']
        vehicle_entities[i].current_real_belong_MBS = real_belong_MBS
        vehicle_entities[i].current_real_belong_RSU = real_belong_RSU
        RSU_entities[real_belong_RSU].serving_vehicles.append(i)
        MBS_entities[real_belong_MBS].serving_vehicles.append(i)
        vehicle_entities[i].current_pred_belong_MBS = list(map(lambda x: x['index'],
                                                               vehicle['trajs'][grand_time_slot][
                                                                   'omegastep_belongMBS']))
        vehicle_entities[i].current_pred_belong_RSU = list(map(lambda x: x['index'],
                                                               vehicle['trajs'][grand_time_slot][
                                                                   'omegastep_belongRSU']))
        pred_belong_RSU = max(vehicle['trajs'][grand_time_slot]['omega_RSU_probability'].items(), key=lambda x: x[1])[0]
        RSU_entities[int(pred_belong_RSU)].pred_serving_vehicles.append(i)
    return "continue"


def count_user_distribution():
    global user_num_counter
    counter = []
    for rsu in RSU_entities:
        counter.append(len(rsu.serving_vehicles))

    user_num_counter.append(counter)


def plot_user_distribution():
    global user_num_counter

    time_steps = len(user_num_counter)  # 时间点数
    # 生成 x 轴的时间序列
    time = list(range(time_steps))
    user_average = []

    # 绘制每个 RSU 的折线
    for rsu_id in range(RSU_NUM):
        user_counts = [user_num_counter[t][rsu_id] for t in range(time_steps)]
        user_average.append(sum(user_counts) / len(user_counts))
        plt.plot(time, user_counts, marker='o', label=f'RSU {rsu_id}')

    # 添加标签和标题
    plt.xlabel('Time Step')
    plt.ylabel('Number of Users')
    plt.title('User Count per RSU Over Time')
    plt.legend()
    plt.grid(True)

    # 显示图像
    plt.show()

    # 绘制柱状图
    rsu_labels = [str(i) for i in range(RSU_NUM)]
    plt.bar(rsu_labels, user_average, color='skyblue')

    # 添加标签和标题
    plt.xlabel('RSU ID')
    plt.ylabel('Average Number of Users')
    plt.title('Average User Count per RSU')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 显示图像
    plt.show()


cloud_entity = v2i_entities.Cloud()
MBS_entities = [v2i_entities.MBS(mbs_caching_memory[i]) for i in range(MBS_NUM)]
RSU_entities = [v2i_entities.RSU(i, rsu_caching_memory[i]) for i in range(RSU_NUM)]
vehicle_entities = [v2i_entities.Vehicle() for _ in range(VEHICLE_NUM)]

grand_time_slot = 0  # 时间片计数器

user_num_counter = []

rsu_loss_sum = 0  # 所有时刻的RSU部分loss之和
mbs_loss_sum = 0  # 所有时刻的MBS部分loss之和

# 大时间片的循环
# 大时间片从1开始，舍弃0时间，因为0时间没有对应的A矩阵
while True:
    # 更新环境（用户位置等），并判断是否达到最后一个时间片需要结束循环
    if grand_tick_pass() == "end":
        break
    count_user_distribution()
    print()
    log_recorder.write("\n" * 7)
    print("-----Grand Time Slot: ", grand_time_slot)
    log_recorder.write(f"-----Grand Time Slot: {grand_time_slot}\n")
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
        rsu.agent.x = utils.discretization(prob_vect)
        rsu.agent.sample_f_g(rsu.pred_serving_vehicles)
        # 根据alpha比例，使用概率向量更新x向量的初始值
        # rsu.agent.update_initial_x(prob_vect)
        # # 小时间片
        # for small_time_slot in range(SMALL_TIME_SLOT_NUM):
        #     print()
        #     log_recorder.write("\n")
        #     print("---Small Time Slot: ", small_time_slot)
        #     log_recorder.write(f"---Small Time Slot: {small_time_slot}\n")
        #     for rsu in RSU_entities:
        #         # 采样loss func: f 和 constraint func: g
        #         rsu.agent.sample_f_g(rsu.pred_serving_vehicles)
        #         # 计算y值
        #         rsu.agent.update_y(rsu.pred_serving_vehicles)
        #     # 这里要拆开的原因是，需要先更新所有rsu的y，然后才能互相communicate更新x
        #
        #     # 这个feasible子集用于存储未被memory constraint终止fine-tuning的rsu集合
        #     RSU_entities_feasible = RSU_entities.copy()
        #     for rsu in RSU_entities:
        #         # 如果没有任何RSU能有足够memory来继续cache，则直接跳出循环
        #         if not RSU_entities_feasible:
        #             print("\n\n---No RSU able to cache more content, terminating fine-tuning. ---\n\n")
        #             log_recorder.write("\n\n---No RSU able to cache more content, terminating fine-tuning. ---\n\n")
        #             break
        #         if rsu not in RSU_entities_feasible:
        #             continue
        #         prev_x = rsu.agent.x  # 保存上一时刻的x，若下一时刻的x不满足约束条件，则回滚x
        #         # 更新p并投影到box，得到下一时刻t+1的决策x
        #         rsu.agent.update_x(A_matrix_list[grand_time_slot - 1], RSU_entities)
        #         # 根据t+1时的x更新lambda
        #         rsu.agent.update_lambda()
        #         # 离散化x
        #         decision_x = utils.discretization(rsu.agent.x)
        #         # 检查内存限制
        #         memory_spent = v2i_entities.constraint_memory(decision_x)
        #         if memory_spent > rsu_caching_memory[rsu.id]:
        #             rsu.agent.x = prev_x
        #             RSU_entities_feasible.remove(rsu)
        print()
        log_recorder.write("\n")
        for rsu in RSU_entities:
            print(f"RSU{rsu.id} Decision x (continuous):", [round(num, 4) for num in rsu.agent.x.tolist()])
            log_recorder.write(
                f"RSU{rsu.id} Decision x (continuous): {[round(num, 4) for num in rsu.agent.x.tolist()]}\n")
        print()
        log_recorder.write("\n")
        rsu_loss = [rsu.agent.loss for rsu in RSU_entities]
        print("RSU Loss (continuous x):", rsu_loss)
        log_recorder.write(f"RSU Loss (continuous x): {rsu_loss}\n")
        print()
        log_recorder.write("\n")
        rsu_total_loss = sum([rsu.agent.loss for rsu in RSU_entities])
        print("RSU total Loss (continuous x):", rsu_total_loss)
        log_recorder.write(f"RSU total Loss (continuous x): {rsu_total_loss}\n")
    for rsu in RSU_entities:
        rsu.agent.check_feasibility(rsu.pred_serving_vehicles)
    # MBS的决策
    if grand_time_slot % 5 == 1:
        mbs_update_time = grand_time_slot // 5  # MBS的决策，每5个grand时间片执行一次
        # dp_loss_sum = 0
        # rollout_loss_sum = 0

        # # 方法1：DP背包问题解法
        # for mbs_id, mbs in enumerate(MBS_entities):
        #     weight = content_size
        #     value = [
        #         mbs_content_popularity[mbs_update_time]['content_popu'][mbs_id].get(str(i), 0) * content_size[i] / v_m2c
        #         for i in range(CONTENT_NUM)]
        #     _, mbs.decision_y = utils.knapsack(weight, value, mbs_caching_memory[mbs_id])
        #     # this_loss, mbs.decision_y = utils.knapsack(weight, value, mbs_caching_memory[mbs_id])
        #     # dp_loss_sum += this_loss
        #     pass

        # start_time = time.time()

        # 方法2：rollout解法
        contents = [i for i in range(CONTENT_NUM)]
        sizes = {i: content_size[i] for i in range(CONTENT_NUM)}
        for mbs_id, mbs in enumerate(MBS_entities):
            capacity = mbs_caching_memory[mbs_id]
            optimizer = rollout.DynamicCacheOptimizer(contents, sizes, capacity)
            predicted_requests_inlist = [mbs_content_popularity[mbs_update_time]['content_popu'][mbs_id].get(str(i), 0)
                                         for i in range(CONTENT_NUM)]
            predicted_requests = {i: predicted_requests_inlist[i] for i in range(CONTENT_NUM)}
            delay_gains_inlist = [content_size[i] / v_m2c for i in range(CONTENT_NUM)]
            delay_gains = {i: delay_gains_inlist[i] for i in range(CONTENT_NUM)}

            # # greedy初始解，添加了content进入概率的参考
            # optimal_cache = rollout.compute_optimal_cache(contents, sizes, capacity, predicted_requests, delay_gains,
            #                                               content_prob=mbs_content_probability[
            #                                                   mbs_update_time]['content_prob'][mbs_id])

            # optimal_cache = rollout.compute_optimal_cache(contents, sizes, capacity, predicted_requests, delay_gains)  # greedy初始解

            optimal_cache = rollout.greedy_plus_local_dp(contents, sizes, capacity, predicted_requests, delay_gains,
                                                         top_k=3)

            # # DP初始解
            # weight = content_size
            # value = [
            #     mbs_content_popularity[mbs_update_time]['content_popu'][mbs_id].get(str(i), 0) * content_size[i] / v_m2c
            #     for i in range(CONTENT_NUM)]
            # _, optimal_cache = utils.knapsack(weight, value, mbs_caching_memory[mbs_id])
            # optimal_cache = set(optimal_cache)

            # # 根据初始解rollout
            # new_cache = optimizer.rollout_step(optimal_cache, predicted_requests, delay_gains)
            # mbs.decision_y = sorted(list(new_cache))

            mbs.decision_y = sorted(list(optimal_cache))

            # # 模拟退火+rollout
            # solution, _ = optimizer.hybrid_optimization(predicted_requests, delay_gains)
            # mbs.decision_y = [i for i in range(CONTENT_NUM) if solution[i] == 1]

            # # 改正后正确的rollout测试
            # env = rollout.MBSEnvironment(contents, sizes, capacity, predicted_requests, delay_gains)
            # agent = rollout.MBSRolloutAgent(env, lambda s: rollout.greedy_policy(s, contents, sizes, predicted_requests, delay_gains,
            #                                                      capacity))
            # initial_cache = rollout2.compute_optimal_cache(contents, sizes, capacity, predicted_requests, delay_gains)
            # best_action = agent.online_decision(initial_cache)
            # best_cache = env.transition(initial_cache, best_action)
            # mbs.decision_y = sorted(list(best_cache))

            # y_list = [1 if i in mbs.decision_y else 0 for i in range(CONTENT_NUM)]
            # rollout_loss_sum += sum([y_list[i] * mbs_content_popularity[mbs_update_time]['content_popu'][mbs_id].get(str(i), 0) * content_size[i] / v_m2c for i in range(CONTENT_NUM)])
            pass

        # end_time = time.time()
        # run_time = end_time - start_time

    print()
    log_recorder.write("\n" * 7)
    print(f"-----!!!END of the grand time slot{grand_time_slot} and calculating: ")
    log_recorder.write(f"-----!!!END of the grand time slot{grand_time_slot} and calculating: \n")
    # 各rsu的决策x（连续形态）
    for rsu in RSU_entities:
        print(f"RSU{rsu.id} Decision x (continuous):", [round(num, 4) for num in rsu.agent.x.tolist()])
        log_recorder.write(f"RSU{rsu.id} Decision x (continuous): {[round(num, 4) for num in rsu.agent.x.tolist()]}\n")
    # 各rsu的决策x（离散形态）
    for rsu in RSU_entities:
        print(f"RSU{rsu.id} Decision x (discrete):", rsu.agent.discrete_x)
        log_recorder.write(f"RSU{rsu.id} Decision x (discrete): {rsu.agent.discrete_x}\n")
    # 各rsu内用户的delay
    local_loss = [v2i_entities.f_func(rsu.agent.discrete_x, rsu.serving_vehicles) for rsu in RSU_entities]
    pred_local_loss = [v2i_entities.f_func(rsu.agent.discrete_x, rsu.pred_serving_vehicles) for rsu in RSU_entities]
    real_log.write(f"{local_loss[54]}\n")
    pred_log.write(f"{pred_local_loss[54]}\n")
    print("RSU Loss (discrete x):", local_loss)
    log_recorder.write(f"RSU Loss (discrete x): {local_loss}\n")
    print()
    log_recorder.write("\n")
    # 各MBS的决策y
    for mbs_id, mbs in enumerate(MBS_entities):
        print(f"MBS{mbs_id} Decision y:", mbs.decision_y)
        log_recorder.write(f"MBS{mbs_id} Decision y: {mbs.decision_y}\n")
    # MBS没有服务到用户造成的附加delay
    users_not_served = []  # 统计各rsu中没有被rsu服务到的用户
    for rsu in RSU_entities:
        for user_id in rsu.serving_vehicles:
            if rsu.agent.discrete_x[get_user_content(user_id)] == 0:
                users_not_served.append(user_id)
    additional_loss = 0
    for user_id in users_not_served:
        content_id = get_user_content(user_id)
        if content_id not in MBS_entities[vehicle_entities[user_id].current_real_belong_MBS].decision_y:
            additional_loss += content_size[content_id] / v_m2c
    print("MBS additional loss (aka. m2c loss):", additional_loss)
    log_recorder.write(f"MBS additional loss (aka. m2c loss): {additional_loss}\n")
    # RSU总delay
    rsu_total_loss = sum(local_loss)
    print("RSU total Loss (discrete x):", rsu_total_loss)
    log_recorder.write(f"RSU total Loss (discrete x): {rsu_total_loss}\n")
    # 总delay
    global_loss = rsu_total_loss + additional_loss
    print("Global Loss (rsu + mbs):", global_loss)
    log_recorder.write(f"Global Loss (rsu + mbs): {global_loss}\n")
    # TODO mbs的内存约束和cost约束
    # 各rsu是否满足内存约束，为布尔量列表
    memory_constraint = [v2i_entities.constraint_memory(rsu.agent.discrete_x) <= rsu.caching_memory for rsu
                         in RSU_entities]
    # 内存约束是否全部满足
    memory_constraint_satisfied = all(memory_constraint)
    print("Memory Constraint Satisfied: ", memory_constraint_satisfied)
    log_recorder.write(f"Memory Constraint Satisfied: {memory_constraint_satisfied}\n")
    # 各rsu是否满足cost约束，为布尔量列表
    local_cost_constraint = [v2i_entities.g_func(rsu.agent.discrete_x, rsu.id) <= 0 for rsu in
                             RSU_entities]
    # 总cost
    global_cost_constraint = sum(
        [v2i_entities.constraint_cost(rsu.agent.discrete_x) for rsu in RSU_entities])
    print("Global Cost Constraint: ", global_cost_constraint)
    log_recorder.write(f"Global Cost Constraint: {global_cost_constraint}\n")
    # 总cost是否小于总cost约束
    global_cost_constraint_satisfied = global_cost_constraint <= sum(local_maximum_cache_cost)
    print("Global Cost Constraint Satisfied: ", global_cost_constraint_satisfied)
    log_recorder.write(f"Global Cost Constraint Satisfied: {global_cost_constraint_satisfied}\n")
    # 所有约束是否全部满足
    constraint_satisfied = all((memory_constraint_satisfied, global_cost_constraint_satisfied))
    print("All Constraint Satisfied: ", constraint_satisfied)
    log_recorder.write(f"All Constraint Satisfied: {constraint_satisfied}\n")
    rsu_loss_sum += rsu_total_loss
    mbs_loss_sum += additional_loss
# plot_user_distribution()
print('\n' * 7)
print(rsu_loss_sum)
print(mbs_loss_sum)
print(rsu_loss_sum + mbs_loss_sum)
