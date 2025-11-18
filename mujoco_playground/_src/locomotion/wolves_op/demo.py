import mujoco
import mujoco.viewer
import glfw
import numpy as np
import pickle
import jax.numpy as jnp
import os
import jax.random
from collections import namedtuple

# Brax imports
from brax.training.agents.ppo import networks as ppo_networks

# ---------------------------------------------------------
# Network Configuration to Match Loaded Policy
# ---------------------------------------------------------
# Based on the error history, the network architecture is definitively:
# Input Size: 49 (nq + nv + 10 sensors + 2 commands)
# Hidden Layers: 512, 256, 128
EXPLICIT_OBSERVATION_SIZE = 49
EXPLICIT_HIDDEN_LAYERS = (512, 256, 128) 

# ---------------------------------------------------------
# Load MuJoCo model
# ---------------------------------------------------------
# Assuming the script runs from a directory containing 'xmls' and 'policy.pkl'
base = os.path.dirname(__file__)
xml_path = os.path.join(base, "xmls", "scene_mjx_feetonly_flat_terrain.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)


# ---------------------------------------------------------
# Load policy parameters
# ---------------------------------------------------------
policy_path = os.path.join(base, "policy.pkl")

try:
    with open(policy_path, "rb") as f:
        policy_params = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Policy file not found at {policy_path}")
    print("Please ensure 'policy.pkl' is in the expected location.")
    exit()

def get_inference_params(loaded_params):
    """
    Attempts to extract the parameter dictionary and re-key it for Brax's 
    make_inference_fn, which expects parameters to be keyed by 0 (normalizer) 
    and 1 (policy).
    """
    
    # 1. Try to extract the parameter dictionary from the common checkpoint tuple structure: (key, dict)
    if isinstance(loaded_params, tuple) and len(loaded_params) > 1 and isinstance(loaded_params[1], dict):
        params_dict = loaded_params[1]
    elif isinstance(loaded_params, dict):
        params_dict = loaded_params
    else:
        # Final fallback, in case it's a raw JAX/Flax structure
        print("Warning: Loaded policy is not a dict or a (key, dict) tuple.")
        return loaded_params
        
    # 2. Check if the dictionary already uses integer keys 0 and 1 (the target format)
    if 0 in params_dict and 1 in params_dict:
        return params_dict

    # 3. If integer keys are missing, we assume the checkpoint uses named keys 
    # (e.g., 'normalizer', 'policy_network') and we must re-key them to 0 and 1.
    normalizer_params = params_dict.get('normalizer')
    policy_params = params_dict.get('policy', params_dict.get('ppo_policy'))
    
    if normalizer_params is not None and policy_params is not None:
        print("Detected named sub-dictionaries ('normalizer', 'policy/ppo_policy'). Re-keying to {0, 1}.")
        # Construct the structure that Brax's make_inference_fn expects
        return {
            0: normalizer_params, 
            1: policy_params      
        }
    
    # 4. Final fallback: If standard named keys aren't found, try to find a single sub-dict named 'params'
    # and recursively check it.
    if 'params' in params_dict and isinstance(params_dict['params'], dict):
        return get_inference_params(params_dict['params'])

    # If all extraction attempts fail, return the highest-level dict we found.
    print("Warning: Could not determine the parameter key structure. Passing highest-level dict.")
    return params_dict


# ---------------------------------------------------------
# Extract Parameters & Build Inference Function
# ---------------------------------------------------------
observation_size = EXPLICIT_OBSERVATION_SIZE # Use the size required by the loaded policy
action_size = model.nu

# Create the network architecture, forcing it to match the loaded weights.
network = ppo_networks.make_ppo_networks(
    observation_size,
    action_size,
    policy_hidden_layer_sizes=EXPLICIT_HIDDEN_LAYERS,
    value_hidden_layer_sizes=EXPLICIT_HIDDEN_LAYERS 
)

# Call the robust parameter extraction
full_params_dict = get_inference_params(policy_params)

inference_fn = None

try:
    print("Attempting to create inference function using 'ppo_networks.make_inference_fn'...")
    make_inference = ppo_networks.make_inference_fn(network)
    # The policy expects the parameters structured as {0: normalizer_params, 1: policy_params}
    inference_fn = make_inference(full_params_dict) 
    print("Success: Policy inference function created.")

except AttributeError:
    print("\n--- CRITICAL ERROR ---")
    print("The 'ppo_networks.make_inference_fn' method is not available in your Brax installation.")
    exit()
except Exception as e:
    print(f"\n--- CRITICAL ERROR ---")
    print(f"An unexpected error occurred during inference function creation. This is likely due to remaining parameter keying issues or mismatch in network shapes: {e}")
    exit()


def policy_fn(obs):
    """
    Applies the inference function returned by make_inference_fn.
    """
    obs = jnp.array(obs)
    # Use a fixed key for inference, as sampling is not needed for deterministic control
    key = jax.random.PRNGKey(0) 
    act, _ = inference_fn(obs, key) 
    return np.array(act)


# ---------------------------------------------------------
# Keyboard handling (FIXED SIGNATURE using *args)
# ---------------------------------------------------------
keys = {}

# Use *args to accept any number of arguments passed by the viewer.
def key_callback(*args): 
    """Handles key presses and releases by updating the global 'keys' dictionary."""
    # The standard GLFW key callback provides 5 arguments.
    if len(args) >= 5:
        # Arguments are typically (window, key, scancode, action, mods)
        key = args[1]
        action = args[3]
    elif len(args) >= 3:
        # A simplified MuJoCo viewer sometimes provides (window, key, action)
        key = args[1]
        action = args[2]
    else:
        return

    # Store True if key is pressed (or repeating), False if released
    keys[key] = (action != glfw.RELEASE)


# ---------------------------------------------------------
# Viewer
# ---------------------------------------------------------
try:
    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
except Exception as e:
    print(f"Failed to launch MuJoCo viewer: {e}")
    exit()


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------
print("Viewer started. Use Arrow Keys to control the robot (Up/Down for velocity, Left/Right for turn). Use Left Shift for sprint.")
while viewer.is_running():

    desired_forward_vel = 0.0
    desired_turn_vel = 0.0

    # --- UPDATED KEY MAPPING TO ARROW KEYS ---
    if keys.get(glfw.KEY_UP):
        desired_forward_vel = 1.0
    if keys.get(glfw.KEY_DOWN):
        desired_forward_vel = -1.0
    if keys.get(glfw.KEY_LEFT):
        desired_turn_vel = 1.0 # Positive turn velocity usually means turning left
    if keys.get(glfw.KEY_RIGHT):
        desired_turn_vel = -1.0 # Negative turn velocity usually means turning right
    # --- END UPDATED KEY MAPPING ---
    
    if keys.get(glfw.KEY_LEFT_SHIFT):
        desired_forward_vel *= 2.0 # Sprint!

    # Observation vector construction
    # The policy requires 49 features. Base obs is (nq + nv + 2 command) = 39. 
    # We require 10 features from sensor data.
    NUM_BASE_OBS = model.nq + model.nv 
    REQUIRED_SENSOR_FEATURES = EXPLICIT_OBSERVATION_SIZE - NUM_BASE_OBS - 2 # 49 - 37 - 2 = 10

    # Get the sensor data, ensuring we take or pad to exactly 10 elements.
    if model.nsensor < REQUIRED_SENSOR_FEATURES:
        # Pad the available sensor data with zeros to reach the required length
        sensor_obs = np.pad(data.sensordata, (0, REQUIRED_SENSOR_FEATURES - model.nsensor), 'constant')
    else:
        # Take the first 10 sensor readings
        sensor_obs = data.sensordata[:REQUIRED_SENSOR_FEATURES]


    # Concatenate all 49 required elements
    obs = np.concatenate([
        # 1. State: qpos + qvel (37 features)
        data.qpos,
        data.qvel,
        # 2. Sensor data (10 features)
        sensor_obs, 
        # 3. Command: desired_forward_vel + desired_turn_vel (2 features)
        np.array([desired_forward_vel, desired_turn_vel])
    ])

    # Apply policy
    action = policy_fn(obs)
    data.ctrl[:] = action

    # Step the simulation
    mujoco.mj_step(model, data)
    
    # Update the viewer (essential for passive viewer)
    viewer.sync()

print("Simulation finished.")