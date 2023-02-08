import os
import cv2
import numpy as np
from PIL import Image
import itertools

class ObjectInGridConfigGenerator:
    def __init__(self, split="train"):
        self.grid_positions = [[-0.1, -0.4], [-0.1, -0.2], [-0.1, 0.0], [-0.1, 0.2], [-0.1, 0.4], \
                               [0.1, -0.4],  [0.1, -0.2],  [0.1, 0.0],  [0.1, 0.2],  [0.1, 0.4]]
        self.combinations = []
        for i in range(len(self.grid_positions)):
            candidates = [(i+j) % len(self.grid_positions) for j in range(1, len(self.grid_positions))]
            for c in itertools.combinations(candidates, 1):
                self.combinations.append([i] + list(c))
            for c in itertools.combinations(candidates, 2):
                self.combinations.append([i] + list(c))
            for c in itertools.combinations(candidates, 3):
                self.combinations.append([i] + list(c))
        permuted_indices = np.random.permutation(len(self.combinations))
        if split == "train":
            self.random_config_idx_iter = iter(permuted_indices[:1000])
        else:
            self.random_config_idx_iter = iter(permuted_indices[1000:])

    def generate_random_config(self, cfg):
        # [Objects]
        object_config = self.combinations[next(self.random_config_idx_iter)]
        cfg.ENV.NUM_PRIMITIVE_OBJECTS = len(object_config)
        cfg.ENV.PRIMITIVE_OBJECT_BASE_POSITION = [self.grid_positions[i] + [cfg.ENV.PRIMITIVE_OBJECT_SIZE / 2] for i in object_config]

        # [Hand]
        hand_model = np.random.choice([
            '20200709-subject-01',
            '20200813-subject-02',
            '20200820-subject-03',
            '20200903-subject-04',
            '20200908-subject-05',
            '20200918-subject-06',
            '20200928-subject-07',
            '20201002-subject-08',
            '20201015-subject-09',
            '20201022-subject-10'
        ])
        hand_side = np.random.choice(['left', 'right'])
        cfg.ENV.MANO_MODEL_FILENAME = os.path.join('gestureIL/objects/data/assets', hand_model + '_' + hand_side, 'mano.urdf')
        hand_pose = np.random.randint(1, 5)
        cfg.ENV.MANO_POSE_FILENAME = os.path.join('gestureIL/objects/data/mano_poses', str(hand_pose) + '_' + hand_side + '.npy')
        # cfg.ENV.MANO_INITIAL_TARGET = (np.array(cfg.ENV.PRIMITIVE_OBJECT_BASE_POSITION[cfg.ENV.PICKED_OBJECT_IDX]) + np.random.uniform(-1, 1, 3) * cfg.ENV.PRIMITIVE_OBJECT_SIZE / 4).tolist()
        cfg.ENV.MANO_INITIAL_TARGET = cfg.ENV.PRIMITIVE_OBJECT_BASE_POSITION[cfg.ENV.PICKED_OBJECT_IDX]
        # distance = np.random.uniform(0.05, 0.15)
        # angle = np.random.uniform(0, 2 * np.pi)
        # cfg.ENV.MANO_INITIAL_BASE = (cfg.ENV.MANO_INITIAL_TARGET + np.array([distance * np.cos(angle), distance * np.sin(angle), np.random.uniform(0.025, 0.075)])).tolist()
        cfg.ENV.MANO_INITIAL_BASE = (np.array(cfg.ENV.MANO_INITIAL_TARGET) + np.array([-0.1, 0.0, 0.05])).tolist()
        # cfg.ENV.MANO_FINAL_TARGET = (np.array([cfg.ENV.TARGET_POSITION_X, cfg.ENV.TARGET_POSITION_Y, cfg.ENV.PRIMITIVE_OBJECT_SIZE / 2]) + np.random.uniform(-1, 1, 3) * cfg.ENV.PRIMITIVE_OBJECT_SIZE / 4).tolist()
        cfg.ENV.MANO_FINAL_TARGET = [cfg.ENV.TARGET_POSITION_X, cfg.ENV.TARGET_POSITION_Y, cfg.ENV.PRIMITIVE_OBJECT_SIZE / 2]
        # distance = np.random.uniform(0.05, 0.15)
        # angle = np.random.uniform(0, 2 * np.pi)
        # cfg.ENV.MANO_FINAL_BASE = (cfg.ENV.MANO_FINAL_TARGET + np.array([distance * np.cos(angle), distance * np.sin(angle), np.random.uniform(0.025, 0.075)])).tolist()
        cfg.ENV.MANO_FINAL_BASE = (np.array(cfg.ENV.MANO_FINAL_TARGET) + np.array([-0.1, 0.0, 0.05])).tolist()
        return cfg


def generate_random_config(cfg):
    # [Target]
    # cfg.ENV.TARGET_POSITION_X, cfg.ENV.TARGET_POSITION_Y = generate_random_position()

    # [Objects]
    cfg.ENV.NUM_PRIMITIVE_OBJECTS = np.random.randint(2, 5)
    base_positions = []
    for i in range(cfg.ENV.NUM_PRIMITIVE_OBJECTS):
        while True:
            candidate_position = generate_random_position()
            conflict = False
            for j in range(i):
                conflict = conflict or abs(candidate_position[0] - base_positions[j][0]) < cfg.ENV.PRIMITIVE_OBJECT_SIZE or abs(candidate_position[1] - base_positions[j][1]) < cfg.ENV.PRIMITIVE_OBJECT_SIZE 
            if not conflict:
                break
        base_positions.append(candidate_position + [cfg.ENV.PRIMITIVE_OBJECT_SIZE / 2])
    cfg.ENV.PRIMITIVE_OBJECT_BASE_POSITION = base_positions

    # [Hand]
    hand_model = np.random.choice([
        '20200709-subject-01',
        '20200813-subject-02',
        '20200820-subject-03',
        '20200903-subject-04',
        '20200908-subject-05',
        '20200918-subject-06',
        '20200928-subject-07',
        '20201002-subject-08',
        '20201015-subject-09',
        '20201022-subject-10'
    ])
    hand_side = np.random.choice(['left', 'right'])
    cfg.ENV.MANO_MODEL_FILENAME = os.path.join('gestureIL/objects/data/assets', hand_model + '_' + hand_side, 'mano.urdf')
    hand_pose = np.random.randint(1, 5)
    cfg.ENV.MANO_POSE_FILENAME = os.path.join('gestureIL/objects/data/mano_poses', str(hand_pose) + '_' + hand_side + '.npy')
    # cfg.ENV.MANO_INITIAL_TARGET = (np.array(cfg.ENV.PRIMITIVE_OBJECT_BASE_POSITION[cfg.ENV.PICKED_OBJECT_IDX]) + np.random.uniform(-1, 1, 3) * cfg.ENV.PRIMITIVE_OBJECT_SIZE / 4).tolist()
    cfg.ENV.MANO_INITIAL_TARGET = cfg.ENV.PRIMITIVE_OBJECT_BASE_POSITION[cfg.ENV.PICKED_OBJECT_IDX]
    # distance = np.random.uniform(0.05, 0.15)
    # angle = np.random.uniform(0, 2 * np.pi)
    # cfg.ENV.MANO_INITIAL_BASE = (cfg.ENV.MANO_INITIAL_TARGET + np.array([distance * np.cos(angle), distance * np.sin(angle), np.random.uniform(0.025, 0.075)])).tolist()
    cfg.ENV.MANO_INITIAL_BASE = (np.array(cfg.ENV.MANO_INITIAL_TARGET) + np.array([-0.1, 0.0, 0.05])).tolist()
    # cfg.ENV.MANO_FINAL_TARGET = (np.array([cfg.ENV.TARGET_POSITION_X, cfg.ENV.TARGET_POSITION_Y, cfg.ENV.PRIMITIVE_OBJECT_SIZE / 2]) + np.random.uniform(-1, 1, 3) * cfg.ENV.PRIMITIVE_OBJECT_SIZE / 4).tolist()
    cfg.ENV.MANO_FINAL_TARGET = [cfg.ENV.TARGET_POSITION_X, cfg.ENV.TARGET_POSITION_Y, cfg.ENV.PRIMITIVE_OBJECT_SIZE / 2]
    # distance = np.random.uniform(0.05, 0.15)
    # angle = np.random.uniform(0, 2 * np.pi)
    # cfg.ENV.MANO_FINAL_BASE = (cfg.ENV.MANO_FINAL_TARGET + np.array([distance * np.cos(angle), distance * np.sin(angle), np.random.uniform(0.025, 0.075)])).tolist()
    cfg.ENV.MANO_FINAL_BASE = (np.array(cfg.ENV.MANO_FINAL_TARGET) + np.array([-0.1, 0.0, 0.05])).tolist()
    return cfg

def generate_random_position():
    distance = np.random.uniform(0.4, 0.6)
    angle = np.random.uniform(0, 3 * np.pi / 4) - 3 * np.pi / 8
    return [float(distance * np.cos(angle)) - 0.6, float(distance * np.sin(angle))]

def get_visual_observation(env, store_dir=None):
    images = env.render_offscreen()
    if store_dir is not None:
        for i in range(len(images)):
            os.makedirs(os.path.join(store_dir, str(i)), exist_ok=True)
            render_file = os.path.join(store_dir, str(i), f'{env.frame:04d}.jpg')
            cv2.imwrite(render_file, images[i][:, :, [2, 1, 0, 3]])
    for i, image in enumerate(images):
        images[i] = Image.fromarray(image, mode="RGBA").convert("RGB")
    return images
