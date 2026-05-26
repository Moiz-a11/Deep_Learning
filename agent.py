
import flappy_bird_gymnasium
import  gymnasium as gym
from dqn import DQN


def run(self,is_training=True,render=False):
    env = gym.make("FlappyBird-v0",render_mode=  "human" if render else None)
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    policy_dqn = DQN(num_states,num_actions).to(device)
    state, _  = env.reset()
    while True:
        action = env.action_space_sample()

        obs, reward, terminated, _, _ = env.step(action)

        if terminated:

            break

    env.close()