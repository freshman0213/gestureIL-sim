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

HORIZONTAL_REACH_THRESHOLD = 0.01
VERTICAL_REACH_THRESHOLD = 0.01
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
        perfect_hand_position = np.array([
            picked_object_position[0] + 0.005,
            picked_object_position[1],
            panda_hand_position[2],
        ])
        randomization = np.random.normal(0, 0.2, 3) * np.array([0.025, 0.025, 0.075])
        self._target_hand_position = perfect_hand_position + randomization
        # self._target_fingers_position = (0.04, 0.04)

    def _move(self, noise_level=0, with_object=False):
        panda_hand_position = self._panda.body.link_state[0, self._panda.LINK_IND_HAND, 0:3].numpy()
        if not np.allclose(panda_hand_position, self._target_hand_position, atol=HORIZONTAL_REACH_THRESHOLD):
            perfect_displacement = self._target_hand_position - panda_hand_position
            noise = np.random.normal(0, 0.2, 3) * noise_level
            noisy_displacement = perfect_displacement + noise
            noisy_displacement[(perfect_displacement * noisy_displacement) < 0] = 0
            self.action = np.concatenate((np.clip((np.around(noisy_displacement / 0.025).astype(int) + 2), 0, 4), [1 if with_object else 0]))
            return True
        else:
            noise = np.random.normal(0, 0.2, 3) * noise_level
            self.action = np.concatenate((np.clip((np.around(noise / 0.025).astype(int) + 2), 0, 4), [1 if with_object else 0]))
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
        if self._execution_phase == 0 and not self._move(noise_level=0.025):
            perfect_hand_position = np.array([
                panda_hand_position[0],
                panda_hand_position[1],
                self._cfg.ENV.PRIMITIVE_OBJECT_SIZE + 0.08
            ])
            randomization = np.random.normal(0, 0.2, 3) * np.array([0, 0, 0.025])
            self._target_hand_position = perfect_hand_position + randomization
            self._execution_phase += 1
        if self._execution_phase == 1 and not self._move(noise_level=np.array([0, 0, 0.025])):
            self._execution_phase += 1
        if self._execution_phase == 2 and not self._move(noise_level=0, with_object=True) and self._object_picked():
            perfect_hand_position = (
                panda_hand_position[0],
                panda_hand_position[1],
                panda_hand_position[2] + 0.2
            )
            randomization = np.random.normal(0, 0.2, 3) * np.array([0, 0, 0.075])
            self._target_hand_position = perfect_hand_position + randomization
            self._execution_phase += 1
        if self._execution_phase == 3 and not self._move(noise_level=np.array([0, 0, 0.025]), with_object=True):
            perfect_hand_position = (
                self._target_position[0],
                self._target_position[1],
                panda_hand_position[2]
            )
            randomization = np.random.normal(0, 0.2, 3) * np.array([0.025, 0.025, 0])
            self._target_hand_position = perfect_hand_position + randomization
            self._execution_phase += 1
        if self._execution_phase == 4 and not self._move(noise_level=np.array([0.025, 0.025, 0]), with_object=True):
            perfect_hand_position = (
                panda_hand_position[0],
                panda_hand_position[1],
                self._cfg.ENV.PRIMITIVE_OBJECT_SIZE + 0.08
            )
            randomization = np.random.normal(0, 0.2, 3) * np.array([0, 0, 0.025])
            self._target_hand_position = perfect_hand_position + randomization
            self._execution_phase += 1
        if self._execution_phase == 5 and not self._move(noise_level=np.array([0, 0, 0.025]), with_object=True):
            self._execution_phase += 1
        if self._execution_phase == 6 and not self._move(noise_level=0) and not self._object_picked():
            perfect_hand_position = (
                panda_hand_position[0],
                panda_hand_position[1],
                panda_hand_position[2] + 0.2
            )
            randomization = np.random.normal(0, 0.2, 3) * np.array([0, 0, 0.075])
            self._target_hand_position = perfect_hand_position + randomization
            self._execution_phase += 1
        if self._execution_phase == 7 and not self._move(noise_level=np.array([0, 0, 0.025])):
            self._execution_phase += 1
        if self._execution_phase == 8:
            self._move()
        return self.action


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