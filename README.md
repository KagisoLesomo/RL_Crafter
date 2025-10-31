# RL_Crafter
For The RL Assignment

# RL_Crafter
For The RL Assignment

To run DQN:

Open the notebook (`crafter.ipynb`)
Each training method (baseline, framestacking, autoencoder etc.) is in its own cell
Simply **run the cell** to train the model — no command line needed


Alt Algorithm = Proximal Policy Optimization (PPO)

### Baseline (no improvements):

python train.py

### Observation Normalisation only:

python train.py --normalise

### Frame Stacking only (recommend 4):

python train.py --frame_stack 4

### Both improvements:

python train.py --normalise --frame_stack 4

### Optionally, change the number of evaluation episodes:

python train.py --eval_episodes 30

### To specify a different output directory or more training steps:

python train.py --outdir logdir/test2 --steps 1000000 --frame_stack 4 --normalise





### Other
Essentially, to run any of the other versions, just:

python train.py --outdir logdir/run1 --steps 10000 --video-fps 30 --video-size 512
