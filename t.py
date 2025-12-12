import json
from matplotlib import pyplot as plt



jnts_file = "act.json"
joint_names_file = "joint_names.json"

jnts = json.loads(open(jnts_file).read())
joint_names = json.loads(open(joint_names_file).read())

print(len(joint_names))
print(len(jnts))
print(len(jnts[0]))

for i in range(len(jnts[0])):
    plt.plot([j[i] for j in jnts])
plt.show()

