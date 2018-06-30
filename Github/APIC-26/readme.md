Reference: https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/a3c/README.md

Install dependencies: `pip install gym[atari]`

In directory `PolicyGradient/a3c`, run 
`./train.py --model_dir /tmp/a3c --env Breakout-v0 --t_max 5 --eval_every 300 --parallelism 8`
to start the training.
