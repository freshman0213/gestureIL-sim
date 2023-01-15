import numpy as np
import easysim
import os
import torch
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d


class MANO:
    def __init__(self, cfg, scene):
        self._cfg = cfg
        self._scene = scene
        self._body = None

    @property
    def body(self):
        return self._body

    def reset(self):
        self._clean()
        self._make()

    def _clean(self):
        if self.body is not None:
            self._scene.remove_body(self.body)
            self._body = None

    def _make(self):
        if self.body is None:
            body = easysim.Body()
            body.name = self._cfg.ENV.MANO_MODEL_FILENAME
            body.geometry_type = easysim.GeometryType.URDF
            body.urdf_file = self._cfg.ENV.MANO_MODEL_FILENAME
            body.use_fixed_base = True
            body.initial_base_position = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

            # Notes: Compute the hand position, rotation
            self.num_timeframes = 10
            linfit = interp1d([0, 1], np.vstack([np.array(self._cfg.ENV.MANO_INITIAL_BASE), np.array(self._cfg.ENV.MANO_FINAL_BASE)]), axis=0)
            self.intermediate_body_translation = linfit(np.arange(0.0, 1+1e-8, 1 / self.num_timeframes)) 

            linfit = interp1d([0, 1], np.vstack([np.array(self._cfg.ENV.MANO_INITIAL_TARGET), np.array(self._cfg.ENV.MANO_FINAL_TARGET)]), axis=0)
            self.intermediate_target = linfit(np.arange(0.0, 1+1e-8, 1 / self.num_timeframes))
            
            mano_side = 'left' if 'left' in self._cfg.ENV.MANO_MODEL_FILENAME else 'right'
            self.intermediate_rotation = np.array([self.get_intrinsic_euler_rotation_angle(mano_side, self.intermediate_target[t], self.intermediate_body_translation[t]) for t in range(self.num_timeframes+1)])
            
            self.body_pose = np.load(self._cfg.ENV.MANO_POSE_FILENAME)
            body.initial_dof_position = np.concatenate((self.intermediate_body_translation[0], self.intermediate_rotation[0], self.body_pose))
            self.next_timeframe = 1

            body.initial_dof_velocity = [0.0] * 51
            if self._cfg.SIM.SIMULATOR == "bullet":
                # Notes: Skin color
                body.link_color = [(0.875, 0.672, 0.410, 1.0)] * 53
            body.link_collision_filter = [self._cfg.ENV.COLLISION_FILTER_MANO] * 53
            body.link_lateral_friction = [5.0] * 53
            body.link_spinning_friction = [5.0] * 53
            body.link_restitution = [0.5] * 53
            body.link_linear_damping = 10.0
            body.link_angular_damping = 10.0
            body.dof_control_mode = easysim.DoFControlMode.POSITION_CONTROL
            body.dof_max_force = (
                self._cfg.ENV.MANO_TRANSLATION_MAX_FORCE
                + self._cfg.ENV.MANO_ROTATION_MAX_FORCE
                + self._cfg.ENV.MANO_JOINT_MAX_FORCE
            )
            body.dof_position_gain = (
                self._cfg.ENV.MANO_TRANSLATION_POSITION_GAIN
                + self._cfg.ENV.MANO_ROTATION_POSITION_GAIN
                + self._cfg.ENV.MANO_JOINT_POSITION_GAIN
            )
            body.dof_velocity_gain = (
                self._cfg.ENV.MANO_TRANSLATION_VELOCITY_GAIN
                + self._cfg.ENV.MANO_ROTATION_VELOCITY_GAIN
                + self._cfg.ENV.MANO_JOINT_VELOCITY_GAIN
            )
            self._scene.add_body(body)
            self._body = body

    def get_intrinsic_euler_rotation_angle(self, mano_side, pointing_position, hand_base_position):
        normalize = lambda x: x / np.linalg.norm(x)

        # Notes: Use rotation to get the hand facing direction in original frame
        original_vector = normalize(np.array([
            pointing_position[0] - hand_base_position[0],
            pointing_position[1] - hand_base_position[1],
            0
        ]))
        target_vector = normalize(np.array([
            pointing_position[0] - hand_base_position[0],
            pointing_position[1] - hand_base_position[1],
            pointing_position[2] - hand_base_position[2]
        ]))
        rotation_axis = normalize(np.cross(original_vector, target_vector))
        rotation_angle = np.arccos(np.inner(original_vector, target_vector))
        rotation = R.from_mrp(rotation_axis * np.tan(rotation_angle / 4))
        hand_facing_direction_original_frame = rotation.apply([0, 0, -1])
        hand_facing_direction_target_frame = np.array([0, -1, 0])

        pointing_direction_original_frame = normalize(np.array([
                pointing_position[0] - hand_base_position[0],
                pointing_position[1] - hand_base_position[1],
                pointing_position[2] - hand_base_position[2]
        ]))
        # Notes: When having no ratation, the left/ right hand points toward the +x/ -x-axis respectively.
        if mano_side == 'left':
            pointing_direction_target_frame = np.array([1, 0, 0]) 
        else:
            pointing_direction_target_frame = np.array([-1, 0, 0])

        rotation, _ = R.align_vectors(
            np.stack((hand_facing_direction_original_frame, pointing_direction_original_frame)),
            np.stack((hand_facing_direction_target_frame, pointing_direction_target_frame))
        )
        return rotation.as_euler("XYZ").tolist()

    def step(self):
        self.body.dof_target_position = np.concatenate((self.intermediate_body_translation[self.next_timeframe], self.intermediate_rotation[self.next_timeframe], self.body_pose)) 
        while not np.allclose(self.body.dof_state[0, :, 0].numpy(), self.body.dof_target_position, atol=0.01):
            return 
        self.next_timeframe += 1

    def finished(self):
        return self.next_timeframe == self.num_timeframes+1