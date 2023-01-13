import gym

from gestureIL.config import get_config_from_args
from gestureIL.policy import ScriptedPolicy

def main():
    cfg = get_config_from_args()
    env = gym.make('GestureILManoPandaEnv-v0', cfg=cfg)

    while True:
        observation = env.reset()

        while not env.mano_hand.finished():
            env.step(None)
        env.switch_phase()

        observation = env.reset()
        policy = ScriptedPolicy(env)
        action = policy.react(observation)
        observation, _, _, _ = env.step(action)

        while not policy.finished():
            action = policy.react(observation)
            observation, _, _, _ = env.step(action)
        env.switch_phase()
    

if __name__ == '__main__':
    main()