import sys
import gym
import numpy as np
import math
import os
import cv2
import time 

from gestureIL.config import get_config_from_args
from gestureIL.policy import ScriptedPolicy

def main():
    # Notes: cfg.ENV is from gestureIL.config, cfg.SIM is from easysim.config
    cfg = get_config_from_args()
    env = gym.make('GestureILPandaEnv', cfg=cfg)

    while True:
        observation = env.reset()

        if cfg.ENV.RENDER_OFFSCREEN:
            store_image(cfg.ENV.RENDER_DIR, env)
        else:
            time.sleep(0.01)

        policy = ScriptedPolicy(env)
        action = policy.react(observation)
        observation, reward, done, info = env.step(action)

        if cfg.ENV.RENDER_OFFSCREEN:
            store_image(cfg.ENV.RENDER_DIR, env)
        else:
            time.sleep(0.01)

        while not policy.finished():
            action = policy.react(observation)
            observation, reward, done, info = env.step(action)

            if cfg.ENV.RENDER_OFFSCREEN:
                store_image(cfg.ENV.RENDER_DIR, env)
            else:
                time.sleep(0.01)
                
        break

def store_image(render_dir, env):
    render_file = os.path.join('/home/iliad/gestureIL/assets/panda_videos', render_dir, '{:06d}.jpg'.format(env.frame))
    cv2.imwrite(render_file, env.render_offscreen()[:, :, [2, 1, 0, 3]])

if __name__ == '__main__':
    main()