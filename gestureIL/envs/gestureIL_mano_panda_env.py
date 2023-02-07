import easysim
import numpy as np
import pybullet

from gestureIL.envs.gestureIL_mano_env import GestureILManoEnv
from gestureIL.envs.gestureIL_panda_env import GestureILPandaEnv
from gestureIL.objects.table import Table
from gestureIL.objects.panda import Panda
from gestureIL.objects.primitive_object import PrimitiveObject
from gestureIL.objects.mano import MANO

class GestureILManoPandaEnv(GestureILManoEnv, GestureILPandaEnv):
    # Notes: This is called by easysim.SimulatorEnv __init__()
    def init(self):
        self._table = Table(self.cfg, self.scene)
        self._panda = Panda(self.cfg, self.scene)
        self._primitive_object = PrimitiveObject(self.cfg, self.scene)
        self._mano_hand = MANO(self.cfg, self.scene)
        # Notes: phase 0 is for MANO hand to move, phase 1 is for panda to move
        self._phase = 0

        if self.cfg.ENV.RENDER_OFFSCREEN:
            self._render_offscreen_init()

    def switch_phase(self):
        self._phase ^= 1
    
    def pre_reset(self, env_ids):
        if self._phase == 0:
            GestureILManoEnv.pre_reset(self, env_ids)
        else:
            self.mano_hand._clean()
            GestureILPandaEnv.pre_reset(self, env_ids)

    def pre_step(self, action):
        if self._phase == 0:
            GestureILManoEnv.pre_step(self, action)
        else:
            GestureILPandaEnv.pre_step(self, action)

    def step(self, action):
        if self._phase == 0:
            return GestureILManoEnv.step(self, action)
        else:
            return GestureILPandaEnv.step(self, action)