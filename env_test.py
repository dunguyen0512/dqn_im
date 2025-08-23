# from spec_env import SpecEnv ### move and guess 
from spec_env_copy import SpecEnv ### terminate if guess is wrong
import numpy as np

# Run this file to test if the environment is working

env = SpecEnv(type='train', seed=None)

obs = env.reset()
done = False

while not done:

    env.render()
    action = env.action_space.sample()
    print("Action taken:", action)
    dir, Y_pred = action % 4, action // 4
    print("Agent moved %s" % (['Up', 'Down','Left','Right'][dir]))
    # print("Agent guessed: %d" % Y_pred)
    
    _, reward, done, info = env.step(action)
    print("Received reward %.1f on step %d" % (reward, env.steps))