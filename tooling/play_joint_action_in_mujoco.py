# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""receives a series of joint commands for the robot in mujoco. Executes the movement + records joint command/state in a json file"""

from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import json

from mujoco_playground._src.locomotion.wolves_op import wolvesop_constants
from mujoco_playground._src.locomotion.wolves_op.base import get_assets

motion = 9
_HERE = epath.Path(__file__).parent
_MOTION_FILE = _HERE / ".." /"json" / f"motion{motion}.json"
_JOINT_NAMES_FILE = _HERE / ".." / "json" / "joint_names.json"
_PREWARMING_STEPS = 100  # how many steps before the actual motion will be executed
_SIM_DT = 0.002
_DELAY_STEPS = 3


class MotionPlayer:
  """ONNX controller for the wolvesOP humanoid."""

  def __init__(
      self,
      default_angles: np.ndarray,
      n_substeps: int,
      motion: list[int],
      joint: int,
      original_qpos: np.ndarray,
      qpos_addr: int,
      qvel_addr: int,
  ):

    self._output_names = ["continuous_actions"]
    
    # ctrl starts at 0
    self._counter = -_PREWARMING_STEPS
    self._n_substeps = n_substeps
    self._motion_index = 0
    
    self._default_angles = default_angles
    self._joint_command = np.zeros((len(motion)), dtype=np.float32)
    self._joint = joint

    # after that we continue to apply the same command, except for the 1 joint that is being moved in the motion
    for i in range(len(motion)):
      self._joint_command[i] = motion[i]
    
    self._original_qpos = original_qpos.copy()
    self._qpos_addr = qpos_addr
    self._qvel_addr = qvel_addr

    self._cmd_buffer = np.zeros(_DELAY_STEPS)
    for i in range(_DELAY_STEPS):
      self._cmd_buffer[i] = self._default_angles[self._joint]

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    global sensor_addr, x, default_angles, action_scale
    
    # initialize the joint to the correct position
    if self._counter < 0:
      data.qpos[self._qpos_addr + 7 + self._joint] = self._default_angles[self._joint]
      data.qvel[self._qvel_addr + 6 + self._joint] = 0.0

    self._counter += 1
    
    # reset position and velocity of the freejoint to the original state -> keep the bot locked in place
    data.qvel[self._qvel_addr : self._qvel_addr + 6] = 0.0
    data.qpos[self._qpos_addr : self._qpos_addr + 7] = self._original_qpos

    for jnt in range(self._default_angles.shape[0]):
      if jnt != self._joint:
        data.qpos[jnt + self._qpos_addr + 7] = self._default_angles[jnt]
        data.qvel[jnt + self._qvel_addr + 6] = 0.0

    if self._counter >= 0 and self._counter % self._n_substeps == 0 and self._motion_index < self._joint_command.shape[0]:
      for i in reversed(range(1, self._cmd_buffer.shape[0])):
        self._cmd_buffer[i] = self._cmd_buffer[i-1]
      self._cmd_buffer[0] = self._joint_command[self._motion_index]

      cmd = self._cmd_buffer[-1]
      data.ctrl[self._joint] = cmd
    
      actions.append(cmd)

      observations.append(data.qpos[7:][self._joint])

      self._motion_index += 1



def load_callback(model=None, data=None):
  mujoco.set_mjcb_control(None)

  model = mujoco.MjModel.from_xml_path(
      wolvesop_constants.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
      assets=get_assets(),
  )

  data = mujoco.MjData(model)

  mujoco.mj_resetDataKeyframe(model, data, 1)

  m = json.loads(open(_MOTION_FILE).read())
  joint_names = json.loads(open(_JOINT_NAMES_FILE).read())
  joint = joint_names.index(m["joint"])

  ctrl_dt = 1/m["rate"]
  n_substeps = int(round(ctrl_dt / _SIM_DT))
  model.opt.timestep = _SIM_DT

  # the controlled joint should be at the same starting position as it was for the real robot
  default_pose = np.array(model.keyframe("home").qpos[7:])
  default_pose[joint] = m["recorded_motion"][0]

  freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "freejoint")
  qpos_adr = model.jnt_qposadr[freejoint_id]
  qvel_adr = model.jnt_dofadr[freejoint_id]
  original_qpos = data.qpos[qpos_adr : qpos_adr + 7].copy()

  policy = MotionPlayer(
      default_angles=default_pose,
      n_substeps=n_substeps,
      motion=m["original_motion"],
      joint=joint,
      original_qpos=original_qpos,
      qpos_addr=qpos_adr,
      qvel_addr=qvel_adr,
  )

  mujoco.set_mjcb_control(policy.get_control)

  return model, data


actions = []
observations = []
sensor_addr = []
jnts = []
default_angles = None
action_scale = None


if __name__ == "__main__":
  headless = True
  if headless:
      model, data = load_callback()

      # simulate until motion is finished
      m = json.loads(open(_MOTION_FILE).read())
      ctrl_dt = 1/m["rate"]
      substeps = ctrl_dt / _SIM_DT
      steps = np.ceil(len(m["original_motion"]) * substeps + _PREWARMING_STEPS).astype(np.int32)
      for _ in range(steps):
          mujoco.mj_step(model, data)
  else:
    viewer.launch(loader=load_callback)

  m = json.loads(open(_MOTION_FILE).read())
  m["original_motion"] = [float(a) for a in actions]
  m["recorded_motion"] = [float(a) for a in observations]
  open(f"json/motion{motion}sim.json", '+w').write(json.dumps(m))


