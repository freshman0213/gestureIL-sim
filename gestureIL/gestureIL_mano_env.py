import easysim
import numpy as np

from gestureIL.table import Table
from gestureIL.panda import Panda
from gestureIL.primitive_object import PrimitiveObject
from gestureIL.mano import MANO

class GestureILManoEnv(easysim.SimulatorEnv):
    # Notes: This is called by easysim.SimulatorEnv __init__()
    def init(self):
        self._table = Table(self.cfg, self.scene)
        self._panda = Panda(self.cfg, self.scene)
        # Notes: Keep the panda arm fixed
        self.panda.body.dof_target_position = self.panda.body.initial_dof_position
        self._primitive_object = PrimitiveObject(self.cfg, self.scene)
        self._mano_hand = MANO(self.cfg, self.scene)

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
        camera = easysim.Camera()
        camera.name = "offscreen_renderer"
        camera.width = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_WIDTH
        camera.height = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_HEIGHT
        camera.vertical_fov = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_VERTICAL_FOV
        camera.near = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_NEAR
        camera.far = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_FAR
        camera.position = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_POSITION
        camera.target = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_TARGET
        camera.up_vector = (0.0, 0.0, 1.0)
        self.scene.add_camera(camera)
        self._camera = camera
    
    def render_offscreen(self):
        if not self.cfg.ENV.RENDER_OFFSCREEN:
            raise ValueError(
                "`render_offscreen()` can only be called when RENDER_OFFSCREEN is set to True"
            )
        return self._camera.color[0].numpy()

    def pre_reset(self, env_ids):
        self.mano_hand.reset()

    def post_reset(self, env_ids):
        self._frame = 0
        return self._get_observation()

    def pre_step(self, action):
        self.mano_hand.step()

    def post_step(self, action):
        self._frame += 1

        observation = self._get_observation()
        reward = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        return observation, reward, done, info
    
    @property
    def frame(self):
        return self._frame

    def _get_observation(self):
        return None
        
    def _get_reward(self):
        return None

    def _get_done(self):
        if np.allclose(self.mano_hand.body.dof_state[0, :, 0], self.mano_hand.body.dof_target_position, atol=3e-3):
            return True
        else:
            return False

    def _get_info(self):
        return {}