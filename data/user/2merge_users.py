import data_read as data
import json

users1 = data.read_json("filtered_users1_revised.json")
omega_steps1 = data.read_json("user_omega_step1_revised.json")

users2 = data.read_json("filtered_users2_revised.json")
omega_steps2 = data.read_json("user_omega_step2_revised.json")

for user in users2:
    user['user_id'] += 50
for user in omega_steps2:
    user['user_id'] += 50

users3 = data.read_json("filtered_users3_revised.json")
omega_steps3 = data.read_json("user_omega_step3_revised.json")

for user in users3:
    user['user_id'] += 50 + 40
for user in omega_steps3:
    user['user_id'] += 50 + 40

print()

users = users1 + users2 + users3
omega_steps = omega_steps1 + omega_steps2 + omega_steps3

f = open("all_filtered_users.json", 'w')
json.dump(users, f)
f.close()

f = open("all_user_omega_step.json", 'w')
json.dump(omega_steps, f)
f.close()
