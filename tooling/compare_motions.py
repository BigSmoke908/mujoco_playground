import json
from matplotlib import pyplot as plt


motion_index = 1

original_motion = json.loads(open(f"json/motion{motion_index}.json").read())
simulated_motion = json.loads(open(f"json/motion{motion_index}sim.json").read())


plt.plot(original_motion["original_motion"], label="cmd on bot")
plt.plot(simulated_motion["original_motion"], label="cmd in sim")
plt.legend(ncol=2, fontsize="small")
plt.show()

