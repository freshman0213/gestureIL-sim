import easysim
import os
import numpy as np
import torch
import pybullet
import math

from scipy.spatial.transform import Rotation as Rot


class Panda:
    _URDF_FILE = os.path.join(
        os.path.dirname(__file__), "data", "assets", "franka_panda", "panda_gripper.urdf"
    )
    _RIGID_SHAPE_COUNT = 11

    LINK_IND_HAND = 8
    LINK_IND_FINGERS = (9, 10)

    def __init__(self, cfg, scene):
        self._cfg = cfg
        self._scene = scene
        # self._control_type = self._cfg.ENV.PANDA_CONTROL_TYPE

        body = easysim.Body()
        body.name = "panda"
        body.geometry_type = easysim.GeometryType.URDF
        body.urdf_file = self._URDF_FILE
        body.use_fixed_base = True
        body.use_self_collision = True
        body.initial_base_position = (
            self._cfg.ENV.PANDA_BASE_POSITION + self._cfg.ENV.PANDA_BASE_ORIENTATION
        )
        body.initial_dof_position = self._cfg.ENV.PANDA_INITIAL_POSITION
        body.initial_dof_velocity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        body.link_collision_filter = [
            self._cfg.ENV.COLLISION_FILTER_PANDA
        ] * self._RIGID_SHAPE_COUNT
        # Notes: The parameters follow the PandaPickAndPlace-V3 of panda-gym
        body.dof_control_mode = easysim.DoFControlMode.POSITION_CONTROL
        # Notes: PandaPickAndPlace-V3 doesn't specify position gain and velocity gain
        body.dof_position_gain = self._cfg.ENV.PANDA_POSITION_GAIN
        body.dof_velocity_gain = self._cfg.ENV.PANDA_VELOCITY_GAIN
        body.dof_max_force = self._cfg.ENV.PANDA_MAX_FORCE
        # Notes: The default spinning friction value is different from the one used in PandaPickAndPlace-V3
        body.link_spinning_friction = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001, 0.001]
        self._scene.add_body(body)
        self._body = body

    @property
    def body(self):
        return self._body

    @property
    def control_type(self):
        return self._control_type

    def step(self, action):
        action = np.clip(action, 0, 5)
        ee_displacement = (action[:3] - 2) * 0.025
        ee_target_position = self.body.link_state[0, self.LINK_IND_HAND, 0:3].numpy() + ee_displacement
        ee_target_orientation = [1, 0, 0, 0]
        target_arm_angles = pybullet.calculateInverseKinematics(
            self.body.contact_id[0], self.LINK_IND_HAND-1, ee_target_position, ee_target_orientation
        )[:7]

        if action[-1] == 0:
            target_fingers_width = 0.08
        else:
            target_fingers_width = 0
        self.body.dof_target_position = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))


        # # Notes: len(action) == 9
        # if self._control_type == "joint":
        #     dof_target_position = action
        # # Notes: len(action) == 8
        # elif self._control_type == "joint_inc":
        #     action = np.clip(action, -1, 1)
        #     # Notes: limit maximum change in position
        #     arm_joint_ctrl = action[:7] * 0.05
        #     target_arm_angles = self.body.dof_state[0, :7, 0] + arm_joint_ctrl
        #     # Notes: limit maximum change in position
        #     fingers_ctrl = action[-1] * 0.2
        #     fingers_width = self.body.dof_state[0, 7, 0] + self.body.dof_state[0, 8, 0]
        #     target_fingers_width = fingers_width + fingers_ctrl
        #     dof_target_position = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        # # Notes: len(action) == 5
        # elif self._control_type == "ee":
        #     ee_target_position = action[:3]
        #     ee_target_orientation = [1, 0, 0, 0]
        #     dof_target_position = pybullet.calculateInverseKinematics(
        #         self.body.contact_id[0], self.LINK_IND_HAND-1, ee_target_position, ee_target_orientation
        #     )
        #     dof_target_position = np.array(dof_target_position)
        #     dof_target_position[7:9] = action[3:]
        # # Notes: len(action) == 4
        # else: # ee_inc
        #     # action = np.clip(action, -100, 100)
        #     # Notes: limit maximum change in position
        #     ee_displacement = action[:3] * 0.0005
        #     ee_target_position = self.body.link_state[0, self.LINK_IND_HAND, 0:3].numpy() + ee_displacement
        #     # Notes: panda-gym mentioned that clipping the height target a great impact on learning
        #     ee_target_position[2] = np.max((0, ee_target_position[2]))
        #     ee_target_orientation = [1, 0, 0, 0]
        #     target_arm_angles = pybullet.calculateInverseKinematics(
        #         self.body.contact_id[0], self.LINK_IND_HAND-1, ee_target_position, ee_target_orientation
        #     )[:7]
        #     # Notes: limit maximum change in position
        #     fingers_ctrl = action[-1] * 0.002
        #     fingers_width = self.body.dof_state[0, 7, 0] + self.body.dof_state[0, 8, 0]
        #     target_fingers_width = fingers_width + fingers_ctrl
        #     dof_target_position = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        # self.body.dof_target_position = dof_target_position