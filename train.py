import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import argparse
import crafter
from shimmy import GymV21CompatibilityV0
import os

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_reward-ppo/0')
parser.add_argument('--steps', type=float, default=5e5)
parser.add_argument('--video-fps', type=int, default=30, help='FPS for video recording')
parser.add_argument('--video-size', type=int, default=512, help='Size of video frames')
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# Wrapper for observation normalisation
class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super(NormalizeObservation, self).__init__(env)
    def observation(self, obs):
        return obs.astype(np.float32) / 255.0

def make_env():
    env = crafter.Env()
    env = crafter.Recorder(
        env,
        args.outdir,
        save_stats=True,
        save_video=True, # Enable video recording
        save_episode=False,
    )
    env = GymV21CompatibilityV0(env=env)
    env = NormalizeObservation(env)
    return env

# The environment must be vectorized for frame stacking
vec_env = DummyVecEnv([make_env])
vec_env = VecFrameStack(vec_env, n_stack=4)  # Second improvement: stack 4 frames per observation

model = PPO(
    "CnnPolicy",
    vec_env,
    verbose=1,
    tensorboard_log=args.outdir,
    learning_rate=1e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=5,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01
)

print(f"Training for {args.steps} steps...")
print(f"Videos will be saved to: {args.outdir}")

model.learn(total_timesteps=int(args.steps), tb_log_name="ppo_crafter")

model.save(f"{args.outdir}/crafter_ppo_model")

print("Training completed!")
print(f"Videos and logs saved to: {args.outdir}")
