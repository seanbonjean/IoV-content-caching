from data import *


def env_tick_pass():
    """
    每个时间片结束后，调用此函数以更新到下一时刻vehicle从属RSU等状态
    """
    global current_time_slot
    current_time_slot += 1
    for i, vehicle in enumerate(trajs):
        belong_MBS = vehicle['trajs'][current_time_slot]['real_belongMBS']
        belong_RSU = vehicle['trajs'][current_time_slot]['real_belongRSU']
        vehicle_entities[i].current_real_belong_MBS = belong_MBS
        vehicle_entities[i].current_real_belong_RSU = belong_RSU
        RSU_entities[belong_RSU].serving_vehicles.append(i)
        MBS_entities[belong_MBS].serving_vehicles.append(i)
        vehicle_entities[i].current_pred_belong_MBS = map(lambda x: x['index'],
                                                          vehicle['trajs'][current_time_slot]['omegastep_belongMBS'])
        vehicle_entities[i].current_pred_belong_RSU = map(lambda x: x['index'],
                                                          vehicle['trajs'][current_time_slot]['omegastep_belongRSU'])


cloud_entity = v2i_entities.Cloud()
MBS_entities = [v2i_entities.MBS() for _ in range(MBS_NUM)]
RSU_entities = [v2i_entities.RSU() for _ in range(RSU_NUM)]
vehicle_entities = [v2i_entities.Vehicle() for _ in range(VEHICLE_NUM)]

current_time_slot = -1  # 时间片计数器

# TODO 算法实现部分
for i, rsu in enumerate(RSU_entities):
    rsu.agent.update_u()
    rsu.agent.sample_f_g(rsu.serving_vehicles, i)

# TODO 计算目标函数、审查约束条件


env_tick_pass()
