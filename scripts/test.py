import os
import gym
import shutil
import numpy as np
import argparse
import torch
from torchvision import transforms

from gestureIL.config import get_cfg
from utils import ObjectInGridConfigGenerator, generate_random_config, get_visual_observation
from gestureIL.policy import LearnedPolicy
from modules.attention_predictor import AttentionActionPredictor
from modules.encoder import MyEncoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main(args):
    cfg = get_cfg()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    
    np.random.seed(cfg.ENV.RANDOM_SEED)
    config_generator = ObjectInGridConfigGenerator(split="test")
    t, num_success = 0, 0
    while t < args.num_trials:
        # cfg = generate_random_config(cfg)
        cfg = config_generator.generate_random_config(cfg)
        env = gym.make('GestureILManoPandaEnv-v0', cfg=cfg)
        env.reset()

        hand_fail = False
        if args.log_dir is not None:
            gestures_1 = get_visual_observation(env, os.path.join(args.log_dir, f'{t:03d}', 'gesture'))
        else:
            gestures_1 = get_visual_observation(env)
        while not env.mano_hand.finished():
            env.step(None)
            if env.frame > 1000:
                hand_fail = True
                break
        if hand_fail:
            if args.log_dir is not None:
                shutil.rmtree(os.path.join(args.log_dir, f'{t:03d}')) 
            env.close()
            continue
        if args.log_dir is not None:
            gestures_2 = get_visual_observation(env, os.path.join(args.log_dir, f'{t:03d}', 'gesture'))
        else:
            gestures_2 = get_visual_observation(env)
        env.switch_phase()

        env.reset()
        if args.model == "AttentionActionPredictor":
            agent = AttentionActionPredictor()
            agent.load_state_dict(torch.load(args.checkpoint_path))
            transform = transforms.Compose([
                # transforms.Resize(112),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            target_transform = lambda x: np.concatenate((x[:2] / 20, x[2:]))
        elif args.model == "MyEncoder":
            agent = MyEncoder()
            agent.load_state_dict(torch.load(args.checkpoint_path))
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        policy = LearnedPolicy(agent, transform, gestures_1, gestures_2)
        if args.log_dir is not None:
            observations = get_visual_observation(env, os.path.join(args.log_dir, f'{t:03d}', 'execution'))
        else:
            observations = get_visual_observation(env)
        action = policy.react(observations)
        action = target_transform(action)
        env.step(action)
        while not policy.finished():
            if args.log_dir is not None:
                observations = get_visual_observation(env, os.path.join(args.log_dir, f'{t:03d}', 'execution'))
            else:
                observations = get_visual_observation(env)
            action = policy.react(observations)
            action = target_transform(action)
            env.step(action)
            if env.frame > 100:
                break
        if np.allclose(env.primitive_object.bodies[cfg.ENV.PICKED_OBJECT_IDX].link_state[0, 0, 0:2], [cfg.ENV.TARGET_POSITION_X, cfg.ENV.TARGET_POSITION_Y], atol=0.03):
            num_success += 1
            print(f"Succeed in trial: {t}!!!")
        env.close()
        t += 1
    print(f"Task success rate: {num_success / t:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--model", type=str, choices=["AttentionActionPredictor", "MyEncoder"], default="AttentionActionPredictor")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--cfg-file", help="path to config file")
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help=(
            """modify config options at the end of the command; use space-separated """
            """"PATH.KEY VALUE" pairs; see src/easysim/config.py for all options"""
        ),
    )
    args = parser.parse_args()
    main(args)
