import abc
import math
import numpy as np
import pybullet
import torch

class Policy(abc.ABC):
    @abc.abstractmethod
    def react(self, observation):
        """ """

    @abc.abstractmethod
    def finished(self):
        """ """


THRESHOLD = 0.01
class ScriptedPolicy(Policy):
    """Continuous Actions"""
    def __init__(self, env):
        self._env = env
        self._cfg = env.cfg
        self._panda = env.panda
        self._picked_object = env.primitive_object.bodies[self._cfg.ENV.PICKED_OBJECT_IDX]
        self._target_position = (self._cfg.ENV.TARGET_POSITION_X, self._cfg.ENV.TARGET_POSITION_Y)

        self._execution_phase = 0
        self._max_movement = 0.05
    
    def _complete_movement(self):
        panda_ee_position = self._panda.body.link_state[0, self._panda.LINK_IND_HAND, 0:3].numpy()[:2]
        return np.allclose(panda_ee_position, self._target_ee_position, atol=THRESHOLD)

    def react(self, observation):
        panda_ee_position = self._panda.body.link_state[0, self._panda.LINK_IND_HAND, 0:3].numpy()
        if self._execution_phase == 0:
            picked_object_position = self._picked_object.link_state[0, 0, 0:3].numpy()
            self._target_ee_position = np.array([
                picked_object_position[0] + 0.005,
                picked_object_position[1],
            ])

            displacement = self._target_ee_position - panda_ee_position[:2]
            distance = np.linalg.norm(displacement)
            if not self._complete_movement():
                return np.concatenate((displacement * self._max_movement / max(self._max_movement, distance), [0]))
            self._execution_phase += 1
            return np.concatenate(([0, 0], [1]))
        if self._execution_phase == 1:
            self._target_ee_position = np.array(self._target_position)
            
            displacement = self._target_ee_position - panda_ee_position[:2]
            distance = np.linalg.norm(displacement)
            if not self._complete_movement():
                return np.concatenate((displacement * self._max_movement / max(self._max_movement, distance), [0]))
            self._execution_phase += 1
            return np.concatenate(([0, 0], [1]))
        if self._execution_phase == 2:
            return np.concatenate(([0, 0], [0]))

    def finished(self):
        return self._execution_phase == 2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LearnedPolicy(Policy):
    def __init__(self, agent, transform, gestures_1, gestures_2):
        self.agent = agent.to(device)
        self.agent.eval()
        self.transform = transform
        self.gestures_1 = gestures_1
        self.gestures_2 = gestures_2
        if self.transform is not None:
            for i, gesture_1 in enumerate(self.gestures_1):
                self.gestures_1[i] = self.transform(gesture_1)
            for i, gesture_2 in enumerate(self.gestures_2):
                self.gestures_2[i] = self.transform(gesture_2)
        for i, gesture_1 in enumerate(self.gestures_1):
            self.gestures_1[i] = torch.unsqueeze(gesture_1, 0).to(device)
        for i, gesture_2 in enumerate(self.gestures_2):
            self.gestures_2[i] = torch.unsqueeze(gesture_2, 0).to(device)

    def react(self, observations):
        if self.transform is not None:
            for i, observation in enumerate(observations):
                observations[i] = self.transform(observation)
        for i, observation in enumerate(observations):
            observations[i] = torch.unsqueeze(observation, 0).to(device)
        with torch.no_grad():
            output = self.agent(*observations, *self.gestures_1, *self.gestures_2)

        if len(output) == 4: # Discrete 3D actions
            x_action, y_action, z_action, gripper_action = output
            x_action = torch.argmax(torch.squeeze(x_action.cpu(), 0)).item()
            y_action = torch.argmax(torch.squeeze(y_action.cpu(), 0)).item()
            z_action = torch.argmax(torch.squeeze(z_action.cpu(), 0)).item()
            gripper_action = torch.argmax(torch.squeeze(gripper_action.cpu(), 0)).item()
            return np.array([x_action, y_action, z_action, gripper_action])
        else: # Continuous 2D actions
            x_action, y_action, gripper_action = output
            x_action = torch.squeeze(x_action.cpu(), 0).item()
            y_action = torch.squeeze(y_action.cpu(), 0).item()
            gripper_action = torch.argmax(torch.squeeze(gripper_action.cpu(), 0)).item()
            return np.array([x_action, y_action, gripper_action])

    def finished(self):
        return False