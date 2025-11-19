from absl import flags
import orbax.checkpoint as ocp
import pathlib
import pickle
import os



__CHECKPOINT_PATH = flags.DEFINE_string("cpp", None, "Provide a path to a checkpoint during training (example: logs/WolvesOPJoystickFlatTerrain-2025.../checkpoints/1000000)")
__ONNX_OUTPUT_PATH = flags.DEFINE_string("onnx", "policy.onnx", "Provide a path to which the generated onnx will be written")
__ENV_NAME = flags.DEFINE_string("env_name", "WolvesOPJoystickFlatTerrain", "Name of the Environemnt this policy was trained with")


