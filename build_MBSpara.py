import json
from route_predict.data_read import read_json
from data import get_user_content, MBS_NUM

user_paras = read_json("route_predict/data/result/parallelograms_0307.json")
pass
step_length = 5

MBS_paras = []  # 第一维：时段；第二维：MBS编号；第3、4维：平行四边形；第五维是content popularity

for period in user_paras:
    this_period_mbs_paras = [[[{} for _ in range(5)] for _ in range(5)] for _ in range(MBS_NUM)]
    for user_id, user_para in enumerate(period['users']):
        content_id = get_user_content(user_id)
        for i, row in enumerate(user_para['belongMBS_para']):
            for j, mbs_id in enumerate(row):
                this_period_mbs_paras[mbs_id][i][j].setdefault(content_id, 0)
                this_period_mbs_paras[mbs_id][i][j][content_id] += 1
    MBS_paras.append({'start_time': period['first_start_time'], 'paras_of_mbs': this_period_mbs_paras})
pass

MBS_content_popu = []  # 第一维：时段；第二维：MBS编号；第三维：content popularity

for period in MBS_paras:
    this_period_popu = [{} for _ in range(MBS_NUM)]
    for mbs_id, para in enumerate(period['paras_of_mbs']):
        for i, row in enumerate(para):
            for j, contents in enumerate(row):
                for content_id, num in contents.items():
                    this_period_popu[mbs_id].setdefault(content_id, 0)
                    this_period_popu[mbs_id][content_id] += num
    MBS_content_popu.append({'start_time': period['start_time'], 'content_popu': this_period_popu})
pass

with open("MBS_content_popu.json", "w") as f:
    json.dump(MBS_content_popu, f)

MBS_content_prob = []  # 第一维：时段；第二维：MBS编号；第三维：content进入该MBS的probability
for period in MBS_paras:
    this_period_prob = [{} for _ in range(MBS_NUM)]
    for mbs_id, para in enumerate(period['paras_of_mbs']):
        for i, row in enumerate(para):
            for j, contents in enumerate(row):
                for content_id in contents.keys():
                    this_period_prob[mbs_id].setdefault(content_id, 0)
                    this_period_prob[mbs_id][content_id] += 1
    for prob_in_mbs in this_period_prob:
        for content in prob_in_mbs.keys():
            prob_in_mbs[content] /= step_length ** 2
    MBS_content_prob.append({'start_time': period['start_time'], 'content_prob': this_period_prob})

with open("MBS_content_prob.json", "w") as f:
    json.dump(MBS_content_prob, f)
