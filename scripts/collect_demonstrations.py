import os
import gym
import cv2
import shutil
import numpy as np

from gestureIL.config import get_config_from_args
from gestureIL.policy import ScriptedPolicy

def main():
    cfg = get_config_from_args()
    
    np.random.seed(cfg.ENV.RANDOM_SEED)
    t = 0
    while t < 200:
        cfg.ENV.TARGET_POSITION_X = np.random.uniform(-0.3, 0.1)
        cfg.ENV.TARGET_POSITION_Y = np.random.uniform(-0.3, 0.3)
        cfg.ENV.NUM_PRIMITIVE_OBJECTS = np.random.randint(2, 4)
        base_positions = []
        for i in range(cfg.ENV.NUM_PRIMITIVE_OBJECTS):
            while True:
                candidate_position = [np.random.uniform(-0.3, 0.1), np.random.uniform(-0.3, 0.3)]
                conflict = False
                for j in range(i):
                    conflict = conflict or abs(candidate_position[0] - base_positions[j][0]) < cfg.ENV.PRIMITIVE_OBJECT_SIZE or abs(candidate_position[1] - base_positions[j][1]) < cfg.ENV.PRIMITIVE_OBJECT_SIZE 
                if not conflict:
                    break
            base_positions.append(candidate_position + [cfg.ENV.PRIMITIVE_OBJECT_SIZE / 2])
        cfg.ENV.PRIMITIVE_OBJECT_BASE_POSITION = base_positions
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
        cfg.ENV.MANO_INITIAL_TARGET = (np.array(cfg.ENV.PRIMITIVE_OBJECT_BASE_POSITION[cfg.ENV.PICKED_OBJECT_IDX]) + np.random.uniform(-1, 1, 3) * cfg.ENV.PRIMITIVE_OBJECT_SIZE / 4).tolist()
        distance = np.random.uniform(0.05, 0.15)
        angle = np.random.uniform(0, 2 * np.pi)
        cfg.ENV.MANO_INITIAL_BASE = (cfg.ENV.MANO_INITIAL_TARGET + np.array([distance * np.cos(angle), distance * np.sin(angle), 0.05])).tolist()
        cfg.ENV.MANO_FINAL_TARGET = (np.array([cfg.ENV.TARGET_POSITION_X, cfg.ENV.TARGET_POSITION_Y, cfg.ENV.PRIMITIVE_OBJECT_SIZE / 2]) + np.random.uniform(-1, 1, 3) * cfg.ENV.PRIMITIVE_OBJECT_SIZE / 4).tolist()
        distance = np.random.uniform(0.05, 0.15)
        angle = np.random.uniform(0, 2 * np.pi)
        cfg.ENV.MANO_FINAL_BASE = (cfg.ENV.MANO_FINAL_TARGET + np.array([distance * np.cos(angle), distance * np.sin(angle), 0.05])).tolist()
        env = gym.make('GestureILManoPandaEnv-v0', cfg=cfg)
        env.reset()

        fail = False
        render_dir_t = os.path.join(cfg.ENV.RENDER_DIR, f'{t:03d}')
        gesture_render_dir = os.path.join(render_dir_t, 'gesture')
        os.makedirs(gesture_render_dir, exist_ok=True)
        for i in range(cfg.ENV.NUM_OFFSCREEN_RENDERER_CAMERA):
            os.makedirs(os.path.join(gesture_render_dir, str(i)), exist_ok=True)
        store_image(gesture_render_dir, env)
        while not env.mano_hand.finished():
            env.step(None)
            if env.frame > 1000:
                fail = True
                break
        if fail:
            shutil.rmtree(render_dir_t) 
            env.close()
            continue
        store_image(gesture_render_dir, env)
        env.switch_phase()

        demonstration_render_dir = os.path.join(render_dir_t, 'demonstration')
        os.makedirs(demonstration_render_dir, exist_ok=True)
        for i in range(cfg.ENV.NUM_OFFSCREEN_RENDERER_CAMERA):
            os.makedirs(os.path.join(demonstration_render_dir, str(i)), exist_ok=True)
        actions = []
        env.reset()
        policy = ScriptedPolicy(env)
        action = policy.react(None)
        store_image(demonstration_render_dir, env)
        actions.append(action)
        env.step(action)
        while not policy.finished():
            action = policy.react(None)
            store_image(demonstration_render_dir, env)
            actions.append(action)
            env.step(action)
            if env.frame > 75:
                fail = True
                break
        fail = fail or not np.allclose(env.primitive_object.bodies[cfg.ENV.PICKED_OBJECT_IDX].link_state[0, 0, 0:2], [cfg.ENV.TARGET_POSITION_X, cfg.ENV.TARGET_POSITION_Y], atol=0.03)
        if fail:
            shutil.rmtree(render_dir_t)
            env.close()
            continue
        actions = np.vstack(actions)
        np.save(os.path.join(demonstration_render_dir, 'actions.npy'), actions)
        env.close()
        t += 1


def store_image(render_dir, env):
    images = env.render_offscreen()
    for i in range(len(images)):
        render_file = os.path.join(render_dir, str(i), f'{env.frame:04d}.jpg')
        cv2.imwrite(render_file, images[i][:, :, [2, 1, 0, 3]])


if __name__ == '__main__':
    main()
