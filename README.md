
### Baseline (no improvements):

bash
python train.py

### Observation Normalisation only:

bash
python train.py --normalise

### Frame Stacking only (recommend 4):

bash
python train.py --frame_stack 4

### Both improvements:

bash
python train.py --normalise --frame_stack 4

### Optionally, change the number of evaluation episodes:

bash
python train.py --eval_episodes 30

### To specify a different output directory or more training steps:

bash
python train.py --outdir logdir/test2 --steps 1000000 --frame_stack 4 --normalise
