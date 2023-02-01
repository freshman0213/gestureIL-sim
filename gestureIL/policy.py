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
                return np.concatenate((displacement * 0.05 / max(0.05, distance), [0]))
            self._execution_phase += 1
            return np.concatenate(([0, 0], [1]))
        if self._execution_phase == 1:
            self._target_ee_position = np.array(self._target_position)
            
            displacement = self._target_ee_position - panda_ee_position[:2]
            distance = np.linalg.norm(displacement)
            if not self._complete_movement():
                return np.concatenate((displacement * 0.05 / max(0.05, distance), [0]))
            self._execution_phase += 1
            return np.concatenate(([0, 0], [1]))
        if self._execution_phase == 2:
            return np.concatenate(([0, 0], [0]))

    def finished(self):
        return self._execution_phase == 2


class LearnedPolicy(Policy):
    def __init__(self, agent, transform, front_gesture_1, side_gesture_1, front_gesture_2, side_gesture_2, device):
        self.agent = agent.to(device)
        self.agent.eval()
        self.transform = transform
        self.front_gesture_1 = front_gesture_1
        self.side_gesture_1 = side_gesture_1
        self.front_gesture_2 = front_gesture_2
        self.side_gesture_2 = side_gesture_2
        self.device = device
        if self.transform is not None:
            self.front_gesture_1 = self.transform(self.front_gesture_1)
            self.side_gesture_1 = self.transform(self.side_gesture_1)
            self.front_gesture_2 = self.transform(self.front_gesture_2)
            self.side_gesture_2 = self.transform(self.side_gesture_2)
        self.front_gesture_1 = torch.unsqueeze(self.front_gesture_1, 0).to(self.device)
        self.side_gesture_1 = torch.unsqueeze(self.side_gesture_1, 0).to(self.device)
        self.front_gesture_2 = torch.unsqueeze(self.front_gesture_2, 0).to(self.device)
        self.side_gesture_2 = torch.unsqueeze(self.side_gesture_2, 0).to(self.device)

    def react(self, observation):
        front_image, side_image = observation
        if self.transform is not None:
            front_image = self.transform(front_image)
            side_image = self.transform(side_image)
        front_image = torch.unsqueeze(front_image, 0).to(self.device)
        side_image = torch.unsqueeze(side_image, 0).to(self.device)
        with torch.no_grad():
            x_action, y_action, z_action, gripper_action = self.agent(front_image, side_image, self.front_gesture_1, self.side_gesture_1, self.front_gesture_2, self.side_gesture_2)
        x_action = torch.argmax(torch.squeeze(x_action.cpu(), 0)).item()
        y_action = torch.argmax(torch.squeeze(y_action.cpu(), 0)).item()
        z_action = torch.argmax(torch.squeeze(z_action.cpu(), 0)).item()
        gripper_action = torch.argmax(torch.squeeze(gripper_action.cpu(), 0)).item()
        return [x_action, y_action, z_action, gripper_action]

    def finished(self):
        return False