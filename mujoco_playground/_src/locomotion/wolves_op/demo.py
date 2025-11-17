import mujoco
import mujoco.viewer
import glfw
import numpy as np
import pickle
import jax.numpy as jnp
from brax.training.agents.ppo import networks as ppo_networks

# ---------------------------------------------------------
# Model laden
# ---------------------------------------------------------
model = mujoco.MjModel.from_xml_path("wolves_op.xml")
data = mujoco.MjData(model)

# ---------------------------------------------------------
# Policy laden
# ---------------------------------------------------------
policy_params = pickle.load(open("policy.pkl", "rb"))

# Policy-Netzwerk neu erstellen (muss identisch zur Trainingskonfiguration sein!)
observation_size = model.nq + model.nv + 2     # qpos + qvel + 2 Steuerinputs
action_size = model.nu

network = ppo_networks.make_ppo_networks(
    observation_size,
    action_size
)

def policy_fn(obs):
    obs = jnp.array(obs)
    act = network.policy_apply(policy_params, obs)
    return np.array(act)

# Keyboard Handling
keys = {}

def key_callback(window, key, scancode, action, mods):
    keys[key] = (action != glfw.RELEASE)

# Viewer
viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)

desired_forward_vel = 0.0
desired_turn_vel = 0.0

while viewer.is_running():

    desired_forward_vel = 0.0
    desired_turn_vel = 0.0

    if keys.get(glfw.KEY_W):
        desired_forward_vel = 1.0
    if keys.get(glfw.KEY_S):
        desired_forward_vel = -1.0
    if keys.get(glfw.KEY_A):
        desired_turn_vel = 1.0
    if keys.get(glfw.KEY_D):
        desired_turn_vel = -1.0
    if keys.get(glfw.KEY_LEFT_SHIFT):
        desired_forward_vel *= 2.0

    obs = np.concatenate([
        data.qpos,
        data.qvel,
        np.array([desired_forward_vel, desired_turn_vel])
    ])

    action = policy_fn(obs)
    data.ctrl[:] = action

    mujoco.mj_step(model, data)
