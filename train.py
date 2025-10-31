import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
import argparse
import crafter
from shimmy import GymV21CompatibilityV0
import os
from collections import defaultdict

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

def make_env(env_id, outdir, use_normalise, record_video=False):
    def _init():
        env = crafter.Env()
        env = crafter.Recorder(
            env,
            outdir,
            save_stats=True,
            save_video=record_video,
            save_episode=False,
        )
        env = GymV21CompatibilityV0(env=env)
        if use_normalise:
            env = NormalizeObservation(env)
        return env
    return _init

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='logdir/baseline')
    parser.add_argument('--steps', type=float, default=50000)
    parser.add_argument('--n_envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--normalise', action='store_true', help='Enable observation normalisation')
    parser.add_argument('--frame_stack', type=int, default=0, help='Number of frames to stack (0=disable)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Create vectorized training environments
    if args.frame_stack > 0:
        vec_env = DummyVecEnv([make_env(i, args.outdir, args.normalise, False) for i in range(args.n_envs)])
        vec_env = VecFrameStack(vec_env, n_stack=args.frame_stack)
    else:
        vec_env = SubprocVecEnv([make_env(i, args.outdir, args.normalise, False) for i in range(args.n_envs)])

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

    # Print configuration
    config_details = []
    if args.normalise:
        config_details.append('Observation Normalisation')
    if args.frame_stack > 0:
        config_details.append(f'Frame Stacking ({args.frame_stack} frames)')
    if not config_details:
        config_details.append('Baseline (No Improvements)')
    
    print(f"\n{'='*70}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Configuration: {', '.join(config_details)}")
    print(f"Training Steps: {int(args.steps)}")
    print(f"Parallel Environments: {args.n_envs}")
    print(f"Output Directory: {args.outdir}")
    print(f"{'='*70}\n")

    model.learn(total_timesteps=int(args.steps), tb_log_name="ppo_crafter")
    model.save(f"{args.outdir}/crafter_ppo_model")

    print("\n✅ Training completed!")

    # ----------- Evaluation with proper environment wrapping --------------
    print(f"\n{'='*70}")
    print(f"EVALUATING AGENT ({args.eval_episodes} episodes)")
    print(f"{'='*70}\n")

    # Create evaluation environment with same wrappers as training
    eval_env = DummyVecEnv([make_env(0, args.outdir + "/eval", args.normalise, record_video=True)])
    if args.frame_stack > 0:
        eval_env = VecFrameStack(eval_env, n_stack=args.frame_stack)

    achievement_counts = defaultdict(int)
    survival_times = []
    cumulative_rewards = []
    max_survival = 0
    min_survival = float('inf')
    total_achievements_per_episode = []

    for ep in range(args.eval_episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0.0
        timesteps = 0
        ep_achievements = set()
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            
            # Extract from vectorized format
            reward = reward[0]
            done = done[0]
            info = info[0]
            
            total_reward += reward
            timesteps += 1
            
            if "achievements" in info:
                for key in ACHIEVEMENT_KEYS:
                    if info["achievements"].get(key, False):
                        ep_achievements.add(key)
        
        num_achievements = len(ep_achievements)
        total_achievements_per_episode.append(num_achievements)
        
        for key in ep_achievements:
            achievement_counts[key] += 1
        
        survival_times.append(timesteps)
        cumulative_rewards.append(total_reward)
        max_survival = max(max_survival, timesteps)
        min_survival = min(min_survival, timesteps)
        
        print(f"Episode {ep+1:2d}/{args.eval_episodes}: Reward={total_reward:6.2f} | Survival={timesteps:4d} | Achievements={num_achievements}")

    # Calculate 
    unlock_rates = {key: achievement_counts[key] / args.eval_episodes for key in ACHIEVEMENT_KEYS}
    unlock_values = [max(rate, 1e-8) for rate in unlock_rates.values()]
    geom_mean_unlock = np.exp(np.mean(np.log(unlock_values)))
    
    # Additional useful metrics
    avg_survival = np.mean(survival_times)
    std_survival = np.std(survival_times)
    avg_reward = np.mean(cumulative_rewards)
    std_reward = np.std(cumulative_rewards)
    max_reward = max(cumulative_rewards)
    min_reward = min(cumulative_rewards)
    avg_achievements = np.mean(total_achievements_per_episode)
    total_unique_achievements = len([k for k, v in unlock_rates.items() if v > 0])

    # Print comprehensive results
    print(f"\n{'='*70}")
    print(f"PERFORMANCE METRICS - {', '.join(config_details)}")
    print(f"{'='*70}")
    

    print(f"  ├─ Geometric Mean of Achievement Unlock Rates: {geom_mean_unlock:.6f}")
    print(f"  ├─ Average Survival Time:                      {avg_survival:.2f} timesteps")
    print(f"  └─ Average Cumulative Reward per Episode:      {avg_reward:.4f}")
    
    # Additional context metrics
    print(f"\n📊 DETAILED STATISTICS:")
    print(f"  ├─ Survival Time:  {avg_survival:.1f} ± {std_survival:.1f} timesteps (range: {min_survival}-{max_survival})")
    print(f"  ├─ Reward:         {avg_reward:.2f} ± {std_reward:.2f} (range: {min_reward:.2f}-{max_reward:.2f})")
    print(f"  ├─ Unique Achievements Unlocked: {total_unique_achievements}/22")
    print(f"  └─ Avg Achievements Per Episode: {avg_achievements:.2f}")
    
    # Achievement unlock rates
    print(f"\n🎯 ACHIEVEMENT UNLOCK RATES (per achievement):")
    unlocked = {k: v for k, v in sorted(unlock_rates.items()) if v > 0}
    locked = {k: v for k, v in sorted(unlock_rates.items()) if v == 0}
    
    if unlocked:
        print(f"  ✅ Unlocked ({len(unlocked)}):")
        for key, rate in unlocked.items():
            count = int(rate * args.eval_episodes)
            print(f"     {key:20s}: {rate:5.1%} ({count}/{args.eval_episodes} episodes)")
    
    if locked:
        print(f"  ❌ Not Unlocked ({len(locked)}): {', '.join(sorted(locked.keys()))}")
    
    print(f"\n💾 Results saved to: {args.outdir}")
    print(f"{'='*70}\n")
