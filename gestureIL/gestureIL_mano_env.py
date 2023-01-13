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
        self.mano_hand.reset()

    def post_reset(self, env_ids):
        self._frame = 0
        return self._get_observation()

    def pre_step(self, action):
        self.mano_hand.step()
        self.panda.step([2, 2, 2, 0])

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
        return None

    def _get_info(self):
        return {}