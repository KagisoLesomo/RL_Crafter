import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import argparse
import crafter
from shimmy import GymV21CompatibilityV0
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_reward-ppo/0')
parser.add_argument('--steps', type=float, default=5e5)
parser.add_argument('--video-fps', type=int, default=30, help='FPS for video recording')
parser.add_argument('--video-size', type=int, default=512, help='Size of video frames')
parser.add_argument('--normalise', action='store_true', help='Enable observation normalisation')
parser.add_argument('--frame_stack', type=int, default=0, help='Number of frames to stack (0=disable)')
parser.add_argument('--eval_episodes', type=int, default=20, help='Number of evaluation episodes')
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

ACHIEVEMENT_KEYS = [
    "plant_tree", "collect_wood", "collect_stone", "collect_coal", "collect_iron", 
    "collect_diamond", "collect_sapling", "collect_fruit", "eat_fruit", "eat_meat", 
    "eat_sapling", "kill_cow", "kill_zombie", "make_axe", "make_pickaxe", "make_sword",
    "make_furnace", "make_workbench", "mine_stone", "mine_coal", "mine_iron", "mine_diamond"
]

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
        save_video=True,
        save_episode=False,
    )
    env = GymV21CompatibilityV0(env=env)
    if args.normalise:
        env = NormalizeObservation(env)
    return env

# Baseline, one or both improvements
if args.frame_stack > 0:
    vec_env = DummyVecEnv([make_env])
    vec_env = VecFrameStack(vec_env, n_stack=args.frame_stack)
    env_to_use = vec_env
else:
    env_to_use = make_env()

model = PPO(
    "CnnPolicy",
    env_to_use,
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
config_details = []
if args.normalise:
    config_details.append('Observation normalisation')
if args.frame_stack > 0:
    config_details.append(f'Frame stacking ({args.frame_stack} frames)')
if not config_details:
    config_details.append('Baseline')
print("Config:", ', '.join(config_details))

model.learn(total_timesteps=int(args.steps), tb_log_name="ppo_crafter")
model.save(f"{args.outdir}/crafter_ppo_model")

print("Training completed!")

# ----------- Evaluation --------------
print("Evaluating agent over", args.eval_episodes, "episodes...")
eval_env = make_env()  # Unvectorized environment for evaluation

achievement_counts = defaultdict(int)
survival_times = []
cumulative_rewards = []

for ep in range(args.eval_episodes):
    obs, info = eval_env.reset()
    done = False
    total_reward = 0.0
    timesteps = 0
    ep_achievements = set()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        total_reward += reward
        timesteps += 1
        # achievements are stored in info["achievements"], if unlocked this step
        if "achievements" in info:
            for key in ACHIEVEMENT_KEYS:
                if info["achievements"].get(key, False):
                    ep_achievements.add(key)
    for key in ep_achievements:
        achievement_counts[key] += 1
    survival_times.append(timesteps)
    cumulative_rewards.append(total_reward)

# Metrics
unlock_rates = {key: achievement_counts[key] / args.eval_episodes for key in ACHIEVEMENT_KEYS}
geom_mean_unlock = np.exp(np.log(np.array(list(unlock_rates.values()) + [1e-8])).mean())  # add small epsilon
avg_survival = np.mean(survival_times)
avg_reward = np.mean(cumulative_rewards)

print("\nAssignment Standard Metrics:")
print("- Achievement Unlock Rates per Achievement:")
for key, rate in sorted(unlock_rates.items()):
    print(f"  {key}: {rate:.2f}")
print(f"- Geometric Mean Unlock Rate: {geom_mean_unlock:.4f}")
print(f"- Avg. Survival Time (timesteps): {avg_survival:.2f}")
print(f"- Avg. Cumulative Reward: {avg_reward:.2f}")
print("\nVideos and logs saved to: {args.outdir}")
