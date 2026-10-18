# TD3 learner

This package implements the continuous-action Twin Delayed Deep Deterministic Policy Gradient learner used by the sequential experiments.
The implementation follows Fujimoto, van Hoof, and Meger and the authors' reference code.
It uses twin critics, clipped target-policy smoothing, delayed actor updates, and Polyak target updates.

The actor maps normalized outputs in `[-1, 1]` to physical regulator ratios.
For bounds 0.90–1.10 p.u., the midpoint is 1.0 and the half-range is 0.1.
Exploration noise has normalized sigma 0.1, or 0.01 p.u.
Target smoothing has normalized sigma 0.2 and clip 0.5, or 0.02 and 0.05 p.u.
The critic and replay buffer always receive physical ratios.

The actor and each critic branch use paper-aligned hidden widths of 400/300 and Adam learning rate `3e-4`.
The previous baseline used the author-code 256/256 recipe.
The replay batch has 256 transitions, gamma is 1.0 for the sequential task, tau is 0.005, and the actor updates every second critic iteration.
The learner samples replay with replacement.

`save_models` writes `Actor_td3` and `Critic_td3`.
`td3.checkpoints.load_agent_for_run` reconstructs inference networks from a compatible `seq_td3` run record and validates checkpoint tensors before loading.
Checkpoints do not include optimizers or replay, so they support inference only and cannot resume training.
