import json
import numpy as np
from etils import epath

_HERE = epath.Path(__file__).parent
_JOINT_NAMES_FILE = _HERE / ".." / "json" / "joint_names.json"


class Animation:

    def __init__(self,
                 default_angles: np.ndarray,
                 joints: list[str],
                 commands: np.ndarray,
                 rate: float) -> None:
        """
        Args:
            default_angles (np.ndarray): should be of shape (n,), with n being the number of total joints
            joints (list[str]): should be the names of the actuacted joints in this animation, length is p
            commands (np.ndarray): the individual joint-commands, should be of shape (x, p,), with x being the number of steps and p the number of actuated joints. These are offsets to the default-angles
            rate (float): rate at which the joint-commands should be replayed
        """
        
        self.default_angles = default_angles.copy()
        self.joints = [jnt for jnt in joints]
        self.commands = commands.copy()
        self.rate = rate
    
    @staticmethod
    def load_from_animation_file(file: str, rate: float=100) -> "Animation":
        joint_names = json.loads(open(_JOINT_NAMES_FILE, "r").read())
        
        raw = json.loads(open(file, "r").read())
        frames = raw["frames"]

        # these should be in the same order as the joints in joint names(!)
        joints = [jnt for jnt in joint_names if jnt in raw["joints"]]
        rate = rate  # -> animations dont really work with a rate -> we just make one up, to work with our MO in the sim
        
        # get all relevant joint positions for every frame
        joint_positions_per_frame = []
        for frame in frames:
            joints_positions_in_frame = []
            
            for jnt in joint_names:
                for jp in frame["joint_positions"]:
                    if jp["name"] == jnt:  # TODO do we need to handle missing joints somehow?
                        joints_positions_in_frame.append(jp["position"])
                        break
            joint_positions_per_frame.append(joints_positions_in_frame)

        # we just use the first frame as starting positions for the joints
        default_angles = joint_positions_per_frame[0]

        # now go through the entire animation, interpolate linearly between different timeframes (is done on servos in reality)        
        # when is every frame finished + how long is this frame (first frame is finished right away)
        frameTimings = [(0, 0)]
        for frame in frames[1:]:
            frameTimings.append((frameTimings[-1][0] + frame["global_runtime_s"], frame["global_runtime_s"]))
        
        joint_commands = []
        print(f"Knee-positions: {[joint_positions_per_frame[i][joint_names.index("knee_l")] for i in range(len(frames))]}")
        t = 0
        dt = 1/rate
        current_frame = 1  # the first frame is used for starting angles -> we start with the second frame
        while t * dt <= frameTimings[-1][0] and current_frame < len(frameTimings):
            try:  # some of our animations (kicks for example) have frames that are supposed to execute immediatly -> just catch the exeception and mark this step as complete
                frame_completion = (t*dt - frameTimings[current_frame-1][0]) / frameTimings[current_frame][1]
            except ZeroDivisionError:
                frame_completion = 1

            joint_offsets = []
            for jnt in range(len(joint_names)):
                joint_after_frame = joint_positions_per_frame[current_frame][jnt]
                joint_before_frame = joint_positions_per_frame[current_frame-1][jnt]
                joint_default = joint_positions_per_frame[0][jnt]

                joint_offsets.append((joint_after_frame-joint_before_frame) * frame_completion - joint_default + joint_before_frame)

            joint_commands.append(joint_offsets)
            t += 1
            # is the current frame completed?
            if t * dt >= frameTimings[current_frame][0]:
                current_frame += 1
        
        return Animation(
            np.array(default_angles, dtype=np.float32),
            joints=joints,
            commands=np.array(joint_commands, dtype=np.float32),
            rate=rate,
        )
        

if __name__ == "__main__":
    animation_file = "json/animations/kick_left.json"

    a = Animation.load_from_animation_file(animation_file)

    from matplotlib import pyplot as plt

    for jnt in json.loads(open(_JOINT_NAMES_FILE).read()):
        jnt_id = a.joints.index(jnt)
        
        plt.plot([cmd[jnt_id] for cmd in a.commands], label=jnt)
    plt.legend(ncol=2, fontsize="small")
    plt.show()

