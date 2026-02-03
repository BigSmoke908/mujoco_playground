import json
from matplotlib import pyplot as plt

medical_to_technical = {
    'LR_HR': 'hip_yaw_r',
    'LR_HAA': 'hip_roll_r',
    'LR_HFE': 'hip_pitch_r',
    'LR_KFE': 'knee_r',
    'LR_FFE': 'ankle_pitch_r',
    'LR_FAA': 'ankle_roll_r',
    'LL_HR': 'hip_yaw_l',
    'LL_HAA': 'hip_roll_l',
    'LL_HFE': 'hip_pitch_l',
    'LL_KFE': 'knee_l',
    'LL_FFE': 'ankle_pitch_l',
    'LL_FAA': 'ankle_roll_l',
}

jnts_file = "act.json"
joint_names_file = "joint_names.json"

jnts = json.loads(open(jnts_file).read())
joint_names = json.loads(open(joint_names_file).read())
joint_names = [medical_to_technical[jnt] for jnt in joint_names]

print(len(joint_names))
print(joint_names)
print(len(jnts))
print(len(jnts[0]))

for i in range(len(jnts[0])):
    if "hip" in joint_names[i]:
        plt.plot([j[i] for j in jnts], label=joint_names[i])

plt.legend(ncol=2, fontsize="small")
plt.show()

