
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
