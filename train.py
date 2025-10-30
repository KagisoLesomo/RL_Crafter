import gymnasium as gym
from stable_baselines3 import PPO
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

# Create log directory
os.makedirs(args.outdir, exist_ok=True)

# Environment setup with recording
env = crafter.Env()
env = crafter.Recorder(
    env, 
    args.outdir,
    save_stats=True,
    save_video=True,  # Enable video recording
    save_episode=False,
)
env = GymV21CompatibilityV0(env=env)

# Create the PPO model
model = PPO(
    "CnnPolicy", 
    env,
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

# Train the model
model.learn(total_timesteps=int(args.steps), tb_log_name="ppo_crafter")

# Save the model
model.save(f"{args.outdir}/crafter_ppo_model")

print("Training completed!")
print(f"Videos and logs saved to: {args.outdir}")