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
        body.dof_target_position = self._cfg.ENV.PANDA_INITIAL_POSITION 
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

    def step(self, action):
        self.body.dof_target_position = action