import json

discard_user_id = [1, 3, 5, 10, 13, 18, 24, 40, 47, 78, 84, ]

f = open("all_filtered_users.json", 'r')
users = json.load(f)
f.close()

f = open("all_user_omega_step.json", 'r')
omega_steps = json.load(f)
f.close()

discarded_users = []
discarded_omega_steps = []

for user in users:
    if user['user_id'] not in discard_user_id:
        discarded_users.append(user)

for user in omega_steps:
    if user['user_id'] not in discard_user_id:
        discarded_omega_steps.append(user)

f = open("discarded_users.json", 'w')
json.dump(discarded_users, f)
f.close()

f = open("discarded_omega_steps.json", 'w')
json.dump(discarded_omega_steps, f)
f.close()
