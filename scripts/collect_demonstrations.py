import os
import gym
import shutil
import numpy as np

from gestureIL.config import get_config_from_args
from utils import ObjectInGridConfigGenerator, generate_random_config, get_visual_observation
from gestureIL.policy import ScriptedPolicy

def main():
    cfg = get_config_from_args()
    
    np.random.seed(cfg.ENV.RANDOM_SEED)
    config_generator = ObjectInGridConfigGenerator()
    t = 0
    while t < 1000:
        # cfg = generate_random_config(cfg)
        cfg = config_generator.generate_random_config(cfg)
        env = gym.make('GestureILManoPandaEnv-v0', cfg=cfg)
        env.reset()

        fail = False
        render_dir_t = os.path.join(cfg.ENV.RENDER_DIR, f'{t:03d}')
        gesture_render_dir = os.path.join(render_dir_t, 'gesture')
        get_visual_observation(env, gesture_render_dir)
        while not env.mano_hand.finished():
            env.step(None)
            if env.frame > 1000:
                fail = True
                break
        if fail:
            shutil.rmtree(render_dir_t) 
            env.close()
            continue
        get_visual_observation(env, gesture_render_dir)
        env.switch_phase()

        demonstration_render_dir = os.path.join(render_dir_t, 'demonstration')
        actions = []
        env.reset()
        policy = ScriptedPolicy(env)
        action = policy.react(None)
        get_visual_observation(env, demonstration_render_dir)
        actions.append(action)
        noise = np.concatenate((np.random.normal(0, 0.5, 2) * 0.01, [0]))
        env.step(action + noise)
        while not policy.finished():
            action = policy.react(None)
            get_visual_observation(env, demonstration_render_dir)
            actions.append(action)
            noise = np.concatenate((np.random.normal(0, 0.5, 2) * 0.01, [0]))
            env.step(action + noise)
            if env.frame > 40:
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


if __name__ == '__main__':
    main()
