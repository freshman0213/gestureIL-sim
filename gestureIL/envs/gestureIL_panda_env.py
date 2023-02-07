import numpy as np
import pybullet

from gestureIL.envs.gestureIL_env import GestureILEnv
from gestureIL.objects.table import Table
from gestureIL.objects.panda import Panda
from gestureIL.objects.primitive_object import PrimitiveObject
from gestureIL.objects.mano import MANO

class GestureILPandaEnv(GestureILEnv):
    # Notes: This is called by easysim.SimulatorEnv __init__()
    def init(self):
        super().init()
        self._table = Table(self.cfg, self.scene)
        self._panda = Panda(self.cfg, self.scene)
        self._primitive_object = PrimitiveObject(self.cfg, self.scene)

    @property
    def table(self):
        return self._table

    @property
    def panda(self):
        return self._panda

    @property
    def primitive_object(self):
        return self._primitive_object

    def pre_reset(self, env_ids):
        pass

    def pre_step(self, action):
        self.panda.step(action)

    def step(self, action):
        action = np.array(action)
        if action.shape[-1] == 4:
            self._panda_discrete_step(action)
        else:
            self._panda_continuous_step(action)

        observation, reward, done, info = self.post_step(action)
        return observation, reward, done, info

    def _panda_discrete_step(self, action):
        action = np.clip(action, 0, 4)
        panda_ee_displacement = (action[:3] - 2) * 0.025
        panda_ee_target_position = self.panda.body.link_state[0, self.panda.LINK_IND_HAND, 0:3].numpy() + panda_ee_displacement
        target_arm_angles = pybullet.calculateInverseKinematics(
            self.panda.body.contact_id[0], self.panda.LINK_IND_HAND-1, panda_ee_target_position, [1, 0, 0, 0]
        )[:7]
        if action[-1] == 0:
            target_fingers_width = 0.08
        else:
            target_fingers_width = 0
        self.pre_step(np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2])))
        # Notes: High-level control runs at 5 Hz
        for _ in range(int(0.2 / self.cfg.SIM.TIME_STEP)):
            self._simulator.step()

    def _object_picked(self):
        finger1_contact, finger2_contact = False, False
        contact = self.contact[0]
        _picked_object = self._primitive_object.bodies[self.cfg.ENV.PICKED_OBJECT_IDX]
        if len(contact) != 0:
            contact_panda2object = contact[
                (contact['body_id_a'] == self._panda.body.contact_id[0])
                & (contact['body_id_b'] == _picked_object.contact_id[0])
            ]
            contact_object2panda = contact[
                (contact['body_id_a'] == _picked_object.contact_id[0])
                & (contact['body_id_b'] == self._panda.body.contact_id[0])
            ]
            contact_object2panda[["body_id_a", "body_id_b"]] = contact_object2panda[["body_id_b", "body_id_a"]]
            contact_object2panda[["link_id_a", "link_id_b"]] = contact_object2panda[["link_id_b", "link_id_a"]]
            contact_object2panda[["position_a_world", "position_b_world"]] = contact_object2panda[["position_b_world", "position_a_world"]]
            contact_object2panda[["position_a_link", "position_b_link"]] = contact_object2panda[["position_b_link", "position_a_link"]]
            contact_object2panda["normal"]["x"] *= -1
            contact_object2panda["normal"]["y"] *= -1
            contact_object2panda["normal"]["z"] *= -1
            contact = np.concatenate((contact_panda2object, contact_object2panda))
            finger1_contact = np.any(contact['link_id_a'] == self._panda.LINK_IND_FINGERS[0])
            finger2_contact = np.any(contact['link_id_a'] == self._panda.LINK_IND_FINGERS[1])
        return finger1_contact and finger2_contact

    def _panda_continuous_step(self, action):
        if action[-1] == 1:
            # Downward Movement
            panda_ee_original_position = self.panda.body.link_state[0, self.panda.LINK_IND_HAND, 0:3].numpy()

            panda_ee_current_position = panda_ee_original_position
            panda_ee_target_position = np.concatenate((panda_ee_original_position[:2], [self.cfg.ENV.PRIMITIVE_OBJECT_SIZE + 0.08]))
            target_arm_angles = pybullet.calculateInverseKinematics(
                self.panda.body.contact_id[0], self.panda.LINK_IND_HAND-1, panda_ee_target_position, [1, 0, 0, 0]
            )[:7]
            self.pre_step(np.concatenate((target_arm_angles, self.panda.body.dof_target_position[-2:])))
            num_steps = 0
            while not np.allclose(panda_ee_current_position, panda_ee_target_position, atol=1e-2):
                num_steps += 1
                self._simulator.step()
                panda_ee_current_position = self.panda.body.link_state[0, self.panda.LINK_IND_HAND, 0:3].numpy()

                if num_steps > 600:
                    break
            
            target_fingers_width = 0 if self.panda.body.dof_target_position[-1] > 0 else 0.08
            self.pre_step(np.concatenate((self.panda.body.dof_target_position[:-2], [target_fingers_width / 2, target_fingers_width / 2])))
            for _ in range(int(0.1 / self.cfg.SIM.TIME_STEP)):
                self._simulator.step()
            if target_fingers_width == 0 and not self._object_picked():
                self.pre_step(np.concatenate((self.panda.body.dof_target_position[:-2], [0.04, 0.04])))
                for _ in range(int(0.1 / self.cfg.SIM.TIME_STEP)):
                    self._simulator.step()

            # Upward Movement
            panda_ee_current_position = self.panda.body.link_state[0, self.panda.LINK_IND_HAND, 0:3].numpy()
            panda_ee_target_position = panda_ee_original_position
            target_arm_angles = pybullet.calculateInverseKinematics(
                self.panda.body.contact_id[0], self.panda.LINK_IND_HAND-1, panda_ee_original_position, [1, 0, 0, 0]
            )[:7] 
            self.pre_step(np.concatenate((target_arm_angles, self.panda.body.dof_target_position[-2:])))
            num_steps = 0
            while not np.allclose(panda_ee_current_position, panda_ee_target_position, atol=1e-2):
                num_steps += 1
                self._simulator.step()
                panda_ee_current_position = self.panda.body.link_state[0, self.panda.LINK_IND_HAND, 0:3].numpy()

                if num_steps > 600:
                    break
        # Planar Movement
        else:
            ##### Continuous Actions #####
            panda_ee_current_position = self.panda.body.link_state[0, self.panda.LINK_IND_HAND, 0:3].numpy()
            panda_ee_target_position = np.concatenate((panda_ee_current_position[:2] + action[:2], [panda_ee_current_position[-1]]))
            panda_ee_target_orientation = [1, 0, 0, 0]
            target_arm_angles = pybullet.calculateInverseKinematics(
                self.panda.body.contact_id[0], self.panda.LINK_IND_HAND-1, panda_ee_target_position, panda_ee_target_orientation
            )[:7]

            self.pre_step(np.concatenate((target_arm_angles, self.panda.body.dof_target_position[-2:])))
            # Notes: High-level control runs at 10 Hz
            for _ in range(int(0.1 / self.cfg.SIM.TIME_STEP)):
                self._simulator.step()

    def _get_observation(self):
        # Notes: This observation space is adapted from the PandaPickAndPlace-v3 of panda-gym
        #   (We discard achieved_goal because it is redundant and desired_goal because we need to infer it from the gesture video)
        ee_position = self.panda.body.link_state[0, self.panda.LINK_IND_HAND, 0:3].numpy() # We are using link frame position instead of center of mass
        ee_velocity = self.panda.body.link_state[0, self.panda.LINK_IND_HAND, 7:10].numpy()
        fingers_width = self.panda.body.dof_state[0, self.panda.LINK_IND_FINGERS[0]-2, 0] + self.panda.body.dof_state[0, self.panda.LINK_IND_FINGERS[1]-2, 0]
        robot_obs = np.concatenate((ee_position, ee_velocity, [fingers_width])).astype(np.float32)

        task_obs = np.array([])
        for i in range(self.cfg.ENV.NUM_PRIMITIVE_OBJECTS):
            object_position = self.primitive_object.bodies[i].link_state[0, 0, 0:3].numpy()
            object_rotation = np.array(self._simulator._p.getEulerFromQuaternion(self.primitive_object.bodies[i].link_state[0, 0, 3:7]))
            # object_velocity = self.primitive_object.bodies[i].link_state[0, 0, 7:10].numpy()
            # object_angular_velocity = self.primitive_object.bodies[i].link_state[0, 0, 10:13].numpy()
            # task_obs = np.concatenate([task_obs, object_position, object_rotation, object_velocity, object_angular_velocity]).astype(np.float32)
            task_obs = np.concatenate([task_obs, object_position, object_rotation]).astype(np.float32)
        return np.concatenate([robot_obs, task_obs])