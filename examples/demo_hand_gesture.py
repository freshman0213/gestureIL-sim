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
    np.random.seed(cfg.ENV.RANDOM_SEED)
    env = gym.make('GestureILManoEnv-v0', cfg=cfg)

    while True:
        env.reset()

        if cfg.ENV.RENDER_OFFSCREEN:
            os.makedirs(cfg.ENV.RENDER_DIR, exist_ok=True)
            for i in range(cfg.ENV.NUM_OFFSCREEN_RENDERER_CAMERA):
                os.makedirs(os.path.join(cfg.ENV.RENDER_DIR, str(i)), exist_ok=True)
            store_image(cfg.ENV.RENDER_DIR, env)

        observation, reward, done, info = env.step(None)

        if cfg.ENV.RENDER_OFFSCREEN:
            store_image(cfg.ENV.RENDER_DIR, env)

        while not done:
            observation, reward, done, info = env.step(None)

            if cfg.ENV.RENDER_OFFSCREEN:
                store_image(cfg.ENV.RENDER_DIR, env)

def store_image(render_dir, env):
    images = env.render_offscreen()
    for i in range(len(images)):
        render_file = os.path.join(render_dir, str(i), '{:06d}.jpg'.format(env.frame))
        cv2.imwrite(render_file, images[i][:, :, [2, 1, 0, 3]])

if __name__ == '__main__':
    main()