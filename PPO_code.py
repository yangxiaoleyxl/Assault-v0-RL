# Effective code with PPO package 
import gym 

from stable_baselines3.common.env_util import make_atari_env 
from stable_baselines3.common.vec_env import VecFrameStack 
from stable_baselines3 import PPO

env = make_atari_env('Assault-v0', n_envs=1, seed=0)  
env = VecFrameStack(env, n_stack=1) 

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='./PPO_log/') 
model.learn(total_timesteps=int(3e4)) 

obs = env.reset() 
obs_ = obs.transpose(3,0,1,2) 

while True:
    action, _states = model.predict(obs_) 
    obs, rewards, dones, info = env.step(action) 
    env.render() 