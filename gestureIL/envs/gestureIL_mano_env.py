import numpy as np

from gestureIL.envs.gestureIL_env import GestureILEnv
from gestureIL.objects.table import Table
from gestureIL.objects.panda import Panda
from gestureIL.objects.primitive_object import PrimitiveObject
from gestureIL.objects.mano import MANO

class GestureILManoEnv(GestureILEnv):
    # Notes: This is called by easysim.SimulatorEnv __init__()
    def init(self):
        super().init()
        self._table = Table(self.cfg, self.scene)
        self._panda = Panda(self.cfg, self.scene)
        self._primitive_object = PrimitiveObject(self.cfg, self.scene)
        self._mano_hand = MANO(self.cfg, self.scene)

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

    def pre_reset(self, env_ids):
        self.mano_hand.reset()

    def pre_step(self, action):
        self.mano_hand.step()