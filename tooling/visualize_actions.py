import json
from matplotlib import pyplot as plt


jnts_file = "json/act.json"
obs_file = "json/obs.json"
joint_names_file = "json/joint_names.json"

jnts = json.loads(open(jnts_file).read())
obs = json.loads(open(obs_file).read())
joint_names = json.loads(open(joint_names_file).read())


def show_single_joint(joint):
    joint_index = joint_names.index(joint)
    
    plt.plot([
        jnt[joint_index] * 0.5
        for jnt in jnts
    ], label=f"{joint} in actionspace")

    # for i in [3, 6, 7, 8, 9, 10]:
    plt.plot([
        observation[joint_index + 9]
        for observation in obs
    ], label=f"{joint} pos in observationspace")



for joint in joint_names:
    show_single_joint(joint)
    plt.legend(ncol=2, fontsize="small")
    plt.show()

