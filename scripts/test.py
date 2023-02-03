import os
import gym
import cv2
import shutil
import numpy as np
import argparse
from PIL import Image
import torch
from torchvision import transforms

from gestureIL.config import get_cfg
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
    t, num_success = 0, 0
    while t < args.num_trials:
        cfg.ENV.TARGET_POSITION_X, cfg.ENV.TARGET_POSITION_Y = generate_random_position()
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
        cfg.ENV.MANO_INITIAL_BASE = (cfg.ENV.MANO_INITIAL_TARGET + np.array([distance * np.cos(angle), distance * np.sin(angle), np.random.uniform(0.025, 0.075)])).tolist()
        cfg.ENV.MANO_FINAL_TARGET = (np.array([cfg.ENV.TARGET_POSITION_X, cfg.ENV.TARGET_POSITION_Y, cfg.ENV.PRIMITIVE_OBJECT_SIZE / 2]) + np.random.uniform(-1, 1, 3) * cfg.ENV.PRIMITIVE_OBJECT_SIZE / 4).tolist()
        distance = np.random.uniform(0.05, 0.15)
        angle = np.random.uniform(0, 2 * np.pi)
        cfg.ENV.MANO_FINAL_BASE = (cfg.ENV.MANO_FINAL_TARGET + np.array([distance * np.cos(angle), distance * np.sin(angle), np.random.uniform(0.025, 0.075)])).tolist()
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
        env.step(action)
        while not policy.finished():
            if args.log_dir is not None:
                observations = get_visual_observation(env, os.path.join(args.log_dir, f'{t:03d}', 'execution'))
            else:
                observations = get_visual_observation(env)
            action = policy.react(observations)
            env.step(action)
            if env.frame > 100:
                break
        if np.allclose(env.primitive_object.bodies[cfg.ENV.PICKED_OBJECT_IDX].link_state[0, 0, 0:2], [cfg.ENV.TARGET_POSITION_X, cfg.ENV.TARGET_POSITION_Y], atol=0.03):
            num_success += 1
        env.close()
        t += 1
    print(f"Task success rate: {num_success / t:.4f}")

def generate_random_position():
    distance = np.random.uniform(0.5, 0.7)
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
