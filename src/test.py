import numpy as np
import gymnasium as gym
import bluesky_gym
bluesky_gym.register_envs()

env = gym.make('MergeEnv-v0', render_mode='human')

obs, info = env.reset()
done = truncated = False
while not (done or truncated):
    action = np.array([0,0]) # Your agent code here
    obs, reward, done, truncated, info = env.step(action)
    print(obs)
