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
from mujoco_playground.experimental.sim2sim.keyboard_gamepad import KeyboardGamepad

motion = 1
_HERE = epath.Path(__file__).parent
_MOTION_FILE = _HERE / ".." /"json" / f"motion{motion}.json"
_JOINT_NAMES_FILE = _HERE / ".." / "json" / "joint_names.json"
_OFFSET = None


class MotionPlayer:
  """ONNX controller for the wolvesOP humanoid."""

  def __init__(
      self,
      default_angles: np.ndarray,
      ctrl_dt: float,
      n_substeps: int,
      motion: list[int],
      joint: int,
      original_qpos: np.ndarray,
      qpos_addr: int,
      qvel_addr: int,
  ):
    global _OFFSET

    self._output_names = ["continuous_actions"]
    
    self._counter = 0
    self._n_substeps = n_substeps
    self._motion_index = 0
    
    rate = 1/ctrl_dt
    self._joint_command = np.zeros((len(motion) + int(rate) * 2, default_angles.shape[0]), dtype=np.float32)
    self._joint = joint
    
    # we put the bot into the default pose for the first 2 seconds (all other joints are constantly controlled to be moved there)
    for i in range(self._joint_command.shape[0]):
      self._joint_command[i][:] = default_angles[:]
    
    # after that we continue to apply the same command, except for the 1 joint that is being moved in the motion
    self._offset = int(rate * 2)
    _OFFSET = self._offset
    for i in range(self._offset, self._offset + len(motion)):
      self._joint_command[i][joint] = motion[i-self._offset]
    
    self._original_qpos = original_qpos.copy()
    self._qpos_addr = qpos_addr
    self._qvel_addr = qvel_addr
    print(self._offset)

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    global sensor_addr, x, default_angles, action_scale
    
    self._counter += 1
    
    # reset position and velocity of the freejoint to the original state -> keep the bot locked in place
    data.qvel[self._qvel_addr : self._qvel_addr + 6] = 0.0
    data.qpos[self._qpos_addr : self._qpos_addr + 7] = self._original_qpos

    if self._counter % self._n_substeps == 0 and self._motion_index < self._joint_command.shape[0]:
      data.ctrl[:] = self._joint_command[self._motion_index]
    
      actions.append(self._joint_command[self._motion_index][self._joint])

      joint_pos = get_joint_positions(model, data)
      observations.append(joint_pos[self._joint])

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
  sim_dt = 0.002
  n_substeps = int(round(ctrl_dt / sim_dt))
  model.opt.timestep = sim_dt

  # the controlled joint should be at the same starting position as it was for the real robot
  default_pose = np.array(model.keyframe("home").qpos[7:])
  default_pose[joint] = m["recorded_motion"][0]

  freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "freejoint")
  qpos_adr = model.jnt_qposadr[freejoint_id]
  qvel_adr = model.jnt_dofadr[freejoint_id]
  original_qpos = data.qpos[qpos_adr : qpos_adr + 7].copy()

  policy = MotionPlayer(
      default_angles=np.array(model.keyframe("home").qpos[7:]),
      ctrl_dt=ctrl_dt,
      n_substeps=n_substeps,
      motion=m["original_motion"],
      joint=joint,
      original_qpos=original_qpos,
      qpos_addr=qpos_adr,
      qvel_addr=qvel_adr,
  )

  mujoco.set_mjcb_control(policy.get_control)

  return model, data


def get_actuated_joint_names(model) -> list[str]:
  # TODO should be put into the training later!

  joint_names = []
  for actuator in range(model.nu):
    joint_id = model.actuator_trnid[actuator][0]
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    joint_names.append(name)
  return joint_names


def get_joint_positions(model: mujoco.MjModel, data: mujoco.MjData):
  return [
    data.qpos[model.jnt_qposadr[j]]
    for j in range(model.njnt)
  ]


actions = []
observations = []
sensor_addr = []
jnts = []
default_angles = None
action_scale = None


if __name__ == "__main__":
  viewer.launch(loader=load_callback)

  m = json.loads(open(_MOTION_FILE).read())
  m["original_motion"] = [float(a) for a in actions[_OFFSET:]]
  m["recorded_motion"] = [float(a) for a in observations[_OFFSET:]]
  open(f"json/motion{motion}sim.json", '+w').write(json.dumps(m))


