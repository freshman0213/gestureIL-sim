import abc
import math
import numpy as np
import pybullet

class Policy(abc.ABC):
    @abc.abstractmethod
    def react(self, observation):
        """ """

    @abc.abstractmethod
    def finished(self):
        """ """

HORIZONTAL_REACH_THRESHOLD = 7e-3
VERTICAL_REACH_THRESHOLD = 7e-3
class ScriptedPolicy(Policy):
    def __init__(self, env):
        self._env = env
        self._cfg = env.cfg
        self._panda = env.panda
        # if self._panda.control_type != "ee":
        #     raise ValueError(f"{self._panda.control_type} is not supported by ScriptedPolicy.")
        self._picked_object = env.primitive_object.bodies[self._cfg.ENV.PICKED_OBJECT_IDX]
        self._target_position = (self._cfg.ENV.TARGET_POSITION_X, self._cfg.ENV.TARGET_POSITION_Y)

        self._execution_phase = 0
        panda_hand_position = self._panda.body.link_state[0, self._panda.LINK_IND_HAND, 0:3].numpy()
        picked_object_position = self._picked_object.link_state[0, 0, 0:3].numpy()
        self._target_hand_position = np.array([
            picked_object_position[0] + 0.005,
            picked_object_position[1],
            panda_hand_position[2],
        ])
        # self._target_fingers_position = (0.04, 0.04)

    def _move(self, with_object=False):
        panda_hand_position = self._panda.body.link_state[0, self._panda.LINK_IND_HAND, 0:3].numpy()
        if not np.allclose(panda_hand_position, self._target_hand_position, atol=HORIZONTAL_REACH_THRESHOLD):
            displacement = self._target_hand_position - panda_hand_position
            self.action = np.clip(np.concatenate((np.around(displacement / 0.005).astype(int) + 10, [10 if not with_object else 9])), 0, 20)
            return True
        else:
            return False

    def _object_picked(self):
        finger1_contact, finger2_contact = False, False
        contact = self._env.contact[0]
        if len(contact) != 0:
            contact_panda2object = contact[
                (contact['body_id_a'] == self._panda.body.contact_id[0])
                & (contact['body_id_b'] == self._picked_object.contact_id[0])
            ]
            contact_object2panda = contact[
                (contact['body_id_a'] == self._picked_object.contact_id[0])
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

    def react(self, observation):
        panda_hand_position = self._panda.body.link_state[0, self._panda.LINK_IND_HAND, 0:3].numpy()
        if self._execution_phase == 0:
            if self._move():
                return self.action
            else:
                self._target_hand_position = np.array([
                    panda_hand_position[0],
                    panda_hand_position[1],
                    self._cfg.ENV.PRIMITIVE_OBJECT_SIZE + 0.08
                ])
                self._execution_phase += 1
        if self._execution_phase == 1:
            if self._move():
                return self.action
            else:
                self._execution_phase += 1
        if self._execution_phase == 2:
            if not self._object_picked():
                return np.array([10, 10, 10, 0])
            else:
                self._target_hand_position = (
                    panda_hand_position[0],
                    panda_hand_position[1],
                    panda_hand_position[2] + 0.2
                )
                self._execution_phase += 1
        if self._execution_phase == 3:
            if self._move(with_object=True):
                return self.action
            else:
                self._target_hand_position = (
                    self._target_position[0],
                    self._target_position[1],
                    panda_hand_position[2]
                )
                self._execution_phase += 1
        if self._execution_phase == 4:
            if self._move(with_object=True):
                return self.action
            else:
                self._target_hand_position = (
                    panda_hand_position[0],
                    panda_hand_position[1],
                    self._cfg.ENV.PRIMITIVE_OBJECT_SIZE + 0.08
                )
                self._execution_phase += 1
        if self._execution_phase == 5:
            if self._move(with_object=True):
                return self.action
            else:
                self._execution_phase += 1
        if self._execution_phase == 6:
            if self._object_picked():
                return np.array([10, 10, 10, 20])
            else:
                self._target_hand_position = (
                    panda_hand_position[0],
                    panda_hand_position[1],
                    panda_hand_position[2] + 0.2
                )
                self._execution_phase += 1
        if self._execution_phase == 7:
            if self._move():
                return self.action
            else:
                self._execution_phase += 1
        if self._execution_phase == 8:
            return np.array([10, 10, 10, 10])


        # panda_hand_position = self._panda.body.link_state[0, self._panda.LINK_IND_HAND, 0:3].numpy()
        # # Approximately horizontal reach
        # if self._execution_phase == 0:
        #     if not np.allclose(panda_hand_position, self._target_hand_position, atol=HORIZONTAL_REACH_THRESHOLD):
        #         return self._target_hand_position + self._target_fingers_position
        #     else:
        #         self._target_hand_position = (
        #             panda_hand_position[0],
        #             panda_hand_position[1],
        #             0.12
        #         )
        #         self._execution_phase += 1
        # # Vertical Reach
        # if self._execution_phase == 1:
        #     if not np.allclose(panda_hand_position, self._target_hand_position, atol=VERTICAL_REACH_THRESHOLD):
        #         return self._target_hand_position + self._target_fingers_position
        #     else:
        #         self._target_fingers_position = (0.00, 0.00)
        #         self._execution_phase += 1
        # # Pick
        # if self._execution_phase == 2:

        #     finger1_contact, finger2_contact = False, False
        #     contact = self._env.contact[0]
        #     if len(contact) != 0:
        #         contact_panda2object = contact[
        #             (contact['body_id_a'] == self._panda.body.contact_id[0])
        #             & (contact['body_id_b'] == self._picked_object.contact_id[0])
        #         ]
        #         contact_object2panda = contact[
        #             (contact['body_id_a'] == self._picked_object.contact_id[0])
        #             & (contact['body_id_b'] == self._panda.body.contact_id[0])
        #         ]
        #         contact_object2panda[["body_id_a", "body_id_b"]] = contact_object2panda[["body_id_b", "body_id_a"]]
        #         contact_object2panda[["link_id_a", "link_id_b"]] = contact_object2panda[["link_id_b", "link_id_a"]]
        #         contact_object2panda[["position_a_world", "position_b_world"]] = contact_object2panda[["position_b_world", "position_a_world"]]
        #         contact_object2panda[["position_a_link", "position_b_link"]] = contact_object2panda[["position_b_link", "position_a_link"]]
        #         contact_object2panda["normal"]["x"] *= -1
        #         contact_object2panda["normal"]["y"] *= -1
        #         contact_object2panda["normal"]["z"] *= -1
        #         contact = np.concatenate((contact_panda2object, contact_object2panda))
        #         finger1_contact = np.any(contact['link_id_a'] == self._panda.LINK_IND_FINGERS[0])
        #         finger2_contact = np.any(contact['link_id_a'] == self._panda.LINK_IND_FINGERS[1])

        #     if finger1_contact and finger2_contact:
        #         self._target_hand_position = (
        #             panda_hand_position[0],
        #             panda_hand_position[1],
        #             panda_hand_position[2] + 0.4
        #         )
        #         self._execution_phase += 1
        #     else:
        #         return self._target_hand_position + self._target_fingers_position
        # # Vertical Move
        # if self._execution_phase == 3:
        #     if not np.allclose(panda_hand_position, self._target_hand_position, atol=VERTICAL_REACH_THRESHOLD):
        #         return self._target_hand_position + self._target_fingers_position
        #     else:
        #         self._target_hand_position = (
        #             self._target_position[0],
        #             self._target_position[1],
        #             panda_hand_position[2]
        #         )
        #         self._execution_phase += 1
        # # Horizontal Move
        # if self._execution_phase == 4:
        #     if not np.allclose(panda_hand_position, self._target_hand_position, atol=HORIZONTAL_REACH_THRESHOLD):
        #         return self._target_hand_position + self._target_fingers_position
        #     else:
        #         self._target_hand_position = (
        #             panda_hand_position[0],
        #             panda_hand_position[1],
        #             0.12
        #         )
        #         self._execution_phase += 1
        # # Vertical Move
        # if self._execution_phase == 5:
        #     if not np.allclose(panda_hand_position, self._target_hand_position, atol=VERTICAL_REACH_THRESHOLD):
        #         return self._target_hand_position + self._target_fingers_position
        #     else:
        #         self._target_fingers_position = (0.04, 0.04)
        #         self._execution_phase += 1
        # # Place
        # if self._execution_phase == 6:
        #     finger1_contact, finger2_contact = False, False
        #     contact = self._env.contact[0]
        #     if len(contact) != 0:
        #         contact_panda2object = contact[
        #             (contact['body_id_a'] == self._panda.body.contact_id[0])
        #             & (contact['body_id_b'] == self._picked_object.contact_id[0])
        #         ]
        #         contact_object2panda = contact[
        #             (contact['body_id_a'] == self._picked_object.contact_id[0])
        #             & (contact['body_id_b'] == self._panda.body.contact_id[0])
        #         ]
        #         contact_object2panda[["body_id_a", "body_id_b"]] = contact_object2panda[["body_id_b", "body_id_a"]]
        #         contact_object2panda[["link_id_a", "link_id_b"]] = contact_object2panda[["link_id_b", "link_id_a"]]
        #         contact_object2panda[["position_a_world", "position_b_world"]] = contact_object2panda[["position_b_world", "position_a_world"]]
        #         contact_object2panda[["position_a_link", "position_b_link"]] = contact_object2panda[["position_b_link", "position_a_link"]]
        #         contact_object2panda["normal"]["x"] *= -1
        #         contact_object2panda["normal"]["y"] *= -1
        #         contact_object2panda["normal"]["z"] *= -1
        #         contact = np.concatenate((contact_panda2object, contact_object2panda))
        #         finger1_contact = np.any(contact['link_id_a'] == self._panda.LINK_IND_FINGERS[0])
        #         finger2_contact = np.any(contact['link_id_a'] == self._panda.LINK_IND_FINGERS[1])

        #     if not finger1_contact and not finger2_contact:
        #         self._target_hand_position = (
        #             panda_hand_position[0],
        #             panda_hand_position[1],
        #             panda_hand_position[2] + 0.4
        #         )
        #         self._execution_phase += 1
        #     else:
        #         return self._target_hand_position + self._target_fingers_position
        # # Leave
        # if self._execution_phase == 7:
        #     if not np.allclose(panda_hand_position, self._target_hand_position, atol=VERTICAL_REACH_THRESHOLD):
        #         return self._target_hand_position + self._target_fingers_position
        #     else:
        #         self._execution_phase += 1
        # # Redudant phase
        # if self._execution_phase == 8:
        #     return self._target_hand_position + self._target_fingers_position         

    def finished(self):
        if self._execution_phase == 8:
            return True
        else:
            return False


class LearnedPolicy(Policy):
    def __init__(self, gesture_video):
        # TODO: Need to loaded the trained imitation learning policy during initialization
        pass