import easysim
import numpy as np
import pybullet

from gestureIL.objects.table import Table
from gestureIL.objects.panda import Panda
from gestureIL.objects.primitive_object import PrimitiveObject
from gestureIL.objects.mano import MANO

class GestureILPandaEnv(easysim.SimulatorEnv):
    # Notes: This is called by easysim.SimulatorEnv __init__()
    def init(self):
        self._table = Table(self.cfg, self.scene)
        self._panda = Panda(self.cfg, self.scene)
        self._primitive_object = PrimitiveObject(self.cfg, self.scene)

        if self.cfg.ENV.RENDER_OFFSCREEN:
            self._render_offscreen_init()

    @property
    def table(self):
        return self._table

    @property
    def panda(self):
        return self._panda

    @property
    def primitive_object(self):
        return self._primitive_object

    @property
    def mano_hand(self):
        return self._mano_hand

    def _render_offscreen_init(self):
        self._cameras = []
        for i in range(self.cfg.ENV.NUM_OFFSCREEN_RENDERER_CAMERA):
            camera = easysim.Camera()
            camera.name = f"offscreen_renderer_{i}"
            camera.width = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_WIDTH
            camera.height = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_HEIGHT
            camera.vertical_fov = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_VERTICAL_FOV
            camera.near = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_NEAR
            camera.far = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_FAR
            camera.position = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_POSITION[i]
            camera.target = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_TARGET[i]
            camera.up_vector = (0.0, 0.0, 1.0)
            self.scene.add_camera(camera)
            self._cameras.append(camera)
    
    def render_offscreen(self):
        if not self.cfg.ENV.RENDER_OFFSCREEN:
            raise ValueError(
                "`render_offscreen()` can only be called when RENDER_OFFSCREEN is set to True"
            )
        return [camera.color[0].numpy() for camera in self._cameras]

    def pre_reset(self, env_ids):
        pass

    def post_reset(self, env_ids):
        self._frame = 0
        return self._get_observation()

    def pre_step(self, action):
        self.panda.step(action)

    def post_step(self, action):
        self._frame += 1
        observation = self._get_observation()
        reward = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        return observation, reward, done, info
    
    def step(self, action):
        action = np.array(action)
        if action.shape[-1] == 4:
            self.panda_discrete_step(action)
        else:
            self.panda_continuous_step(action)

        observation, reward, done, info = self.post_step(action)
        return observation, reward, done, info

    def panda_discrete_step(self, action):
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

    def panda_continuous_step(self, action):
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
            
            # Manipulation
            target_fingers_width = 0 if self.panda.body.dof_target_position[-1] > 0 else 0.08
            self.pre_step(np.concatenate((self.panda.body.dof_target_position[:-2], [target_fingers_width / 2, target_fingers_width / 2])))
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

    @property
    def frame(self):
        return self._frame

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
        
    def _get_reward(self):
        return None

    def _get_done(self):
        return False

    def _get_info(self):
        return {}