# Copyright 2025 Ostfalia Team
# Custom Export Script: Orbax/Pickle to ONNX for WolvesOP
# Compatible with JAX, Orbax, and NumPy 2.0

import os
import argparse
import pickle
import jax.numpy as jp
import orbax.checkpoint as ocp
import tf2onnx
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from mujoco_playground import locomotion
from mujoco_playground.config import locomotion_params

# --- FIX FÜR NUMPY 2.0 ---
# tf2onnx nutzt intern 'np.cast', was in NumPy 2.0 entfernt wurde.
# Wir patchen es hier temporär, damit der Exporter nicht abstürzt.
if not hasattr(np, "cast"):
    class CastMap:
        def __getitem__(self, type_):
            return lambda x: np.asarray(x, dtype=type_)
    np.cast = CastMap()
    print("Info: NumPy 2.0 Compatibility Patch applied.")
# -------------------------

# Fix für Mac M4 / Metal
os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def load_checkpoint(path):
    """Lädt den Checkpoint intelligent (egal ob Datei oder Orbax-Ordner)."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        print(f"Detected Orbax checkpoint folder: {path}")
        checkpointer = ocp.PyTreeCheckpointer()
        return checkpointer.restore(path)
    else:
        print(f"Detected legacy Pickle file: {path}")
        with open(path, 'rb') as f:
            return pickle.load(f)

def get_params_robust(params):
    """Extrahiert Mean, Std und Policy-Weights aus beliebigen Strukturen."""
    
    # 1. Top-Level Struktur entpacken
    if isinstance(params, dict) and "normalizer_params" in params:
        norm_params = params["normalizer_params"]
        pol_params = params["policy_params"]
    elif isinstance(params, (list, tuple)):
        norm_params = params[0]
        pol_params = params[1]
    else:
        norm_params = params
        pol_params = params

    # 2. Hilfsfunktion: Wert suchen
    def find_val(container, candidate_keys):
        for key in candidate_keys:
            if isinstance(container, dict):
                if key in container: return container[key]
            else:
                if hasattr(container, key): return getattr(container, key)
        
        if isinstance(container, dict) and "state" in container:
             return find_val(container["state"], candidate_keys)
        return None

    # Mean finden
    mean = find_val(norm_params, ["mean"])
    if mean is None:
        debug_keys = norm_params.keys() if isinstance(norm_params, dict) else dir(norm_params)
        raise ValueError(f"Konnte 'mean' nicht finden. Verfügbar: {debug_keys}")

    # Std finden
    std = find_val(norm_params, ["std"])
    if std is None:
        var = find_val(norm_params, ["variance", "var", "summed_variance"])
        if var is not None:
            print("Info: 'std' nicht gefunden, berechne aus 'variance'.")
            if isinstance(var, dict):
                std = {k: jp.sqrt(v + 1e-8) for k, v in var.items()}
            else:
                std = jp.sqrt(var + 1e-8)
        else:
            debug_keys = norm_params.keys() if isinstance(norm_params, dict) else dir(norm_params)
            raise ValueError(f"Konnte weder 'std' noch 'variance' finden. Verfügbar: {debug_keys}")
    
    return mean, std, pol_params

def make_policy_network(param_size, mean_std, hidden_layer_sizes, activation=tf.nn.swish):
    """Erstellt das Tensorflow-Modell."""
    class MLP(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.mean = tf.Variable(mean_std[0], trainable=False, dtype=tf.float32)
            self.std = tf.Variable(mean_std[1], trainable=False, dtype=tf.float32)
            self.mlp = tf.keras.Sequential()
            for size in hidden_layer_sizes:
                self.mlp.add(layers.Dense(size, activation=activation, kernel_initializer="lecun_uniform"))
            self.mlp.add(layers.Dense(param_size, kernel_initializer="lecun_uniform"))

        def call(self, inputs):
            x = (inputs - self.mean) / self.std
            logits = self.mlp(x)
            loc, _ = tf.split(logits, 2, axis=-1)
            return tf.tanh(loc)
            
    return MLP()

def main(args):
    print(f"--- Starting Export for {args.env_name} ---")
    
    # Config laden
    env_cfg = locomotion.get_default_config(args.env_name)
    env = locomotion.load(args.env_name, config=env_cfg)
    ppo_params = locomotion_params.brax_ppo_config(args.env_name)
    
    # Dimensionen holen
    if hasattr(env, "observation_size") and hasattr(env.observation_size, "keys"):
         obs_size = env.observation_size["state"][0]
    elif isinstance(env.observation_size, int):
         obs_size = env.observation_size
    else:
         obs_size = env.observation_size[0]
         
    act_size = env.action_size
    print(f"Observation Size: {obs_size}, Action Size: {act_size}")

    # Parameter laden
    raw_params = load_checkpoint(args.checkpoint)
    mean, std, policy_params = get_params_robust(raw_params)

    # Dictionary unpacking
    if isinstance(mean, dict):
        if "state" in mean: mean = mean["state"]
        else: mean = mean[list(mean.keys())[0]]

    if isinstance(std, dict):
        if "state" in std: std = std["state"]
        else: std = std[list(std.keys())[0]]

    # Modell bauen
    mean_std_tf = (tf.convert_to_tensor(mean), tf.convert_to_tensor(std))
    tf_model = make_policy_network(
        param_size=act_size * 2,
        mean_std=mean_std_tf,
        hidden_layer_sizes=ppo_params.network_factory.policy_hidden_layer_sizes
    )
    
    # Init
    dummy_input = tf.zeros((1, obs_size))
    tf_model(dummy_input)

    # Gewichte übertragen
    print("Transferring weights...")
    layer_idx = 0
    if "params" in policy_params: mlp_base = policy_params["params"]
    else: mlp_base = policy_params

    if "MLP_0" in mlp_base: mlp_params = mlp_base["MLP_0"]
    else: mlp_params = mlp_base[list(mlp_base.keys())[0]]

    layer_names = sorted(mlp_params.keys())
    for name in layer_names:
        l_params = mlp_params[name]
        if not isinstance(l_params, dict) or 'kernel' not in l_params: continue

        tf_layer = tf_model.mlp.layers[layer_idx]
        kernel = np.array(l_params['kernel'])
        bias = np.array(l_params['bias'])
        
        tf_layer.set_weights([kernel, bias])
        print(f"Layer {name} populated: {kernel.shape}")
        layer_idx += 1

    # Export
    print(f"Exporting to {args.output}...")
    spec = [tf.TensorSpec(shape=(1, obs_size), dtype=tf.float32, name="obs")]
    tf_model.output_names = ['continuous_actions']
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tf2onnx.convert.from_keras(tf_model, input_signature=spec, opset=11, output_path=args.output)
    print(f"✅ Success! Saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="policy.onnx")
    parser.add_argument("--env-name", type=str, default="WolvesOPJoystickFlatTerrain")
    args = parser.parse_args()
    main(args)