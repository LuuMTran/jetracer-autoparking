# Recurrent PPO with human-scored rewards

The physical loop keeps the camera and car driver open, collects five scored
episodes, performs one PPO update, saves the next numbered checkpoint, and
returns to the ready prompt.

## Trainable parameters

The PPO policy starts from the imitation policy. MobileNetV2, the 256-dimensional
projection, and the 64-unit GRU remain frozen. PPO trains the actor MLP, value
MLP, and two log standard deviations. Rollouts therefore store the frozen
64-dimensional recurrent feature rather than raw images.

The actor is a tanh-squashed diagonal Gaussian. `follow` uses its mean and does
not collect data or update the model.

## Default parameters

| Parameter | Value |
|---|---:|
| Discount factor | 0.99 |
| GAE lambda | 0.95 |
| Clip ratio | 0.20 |
| Learning rate | 0.00003 |
| Update epochs | 4 |
| Value weight | 0.5 |
| Entropy weight | 0.001 |
| Gradient norm limit | 1.0 |
| Initial exploration standard deviation | 0.08 |
| Exploration standard-deviation range | 0.02 to 0.30 |
| Minibatch size | 128 transitions |
| Episodes per update | 5 |
| Control rate | 10 Hz |
| Maximum episode length | 400 actions |

`sequence_length=16` and `sequences_per_batch=8` currently determine the PPO
minibatch size of 128. PPO does not unroll the frozen GRU during updates.

## Rewards

Each applied action receives the configured `-0.002` step cost. After the car is
stopped, enter a terminal reward from `-1` to `+1`. Add `t` after the number to
mark an unfinished truncation that retains value bootstrapping. `D` discards the
episode, and Escape exits without saving it.

The stop head only ends an attempt. It does not assign a positive reward.

## Commands

```bash
python3 05_online_ppo.py init \
  --imitation-checkpoint runs/imitation/best.pt \
  --config ppo_config.json --output runs/ppo --device cpu

python3 05_online_ppo.py inspect \
  --checkpoint runs/ppo/ppo_0000.pt --device cpu

python3 05_online_ppo.py physical-loop \
  --checkpoint runs/ppo --run-dir runs/ppo \
  --device cuda --fp16-encoder \
  --stop-head runs/imitation/stop_head.pt \
  --episodes-per-update 5 --max-frame-age-ms 500

python3 05_online_ppo.py follow \
  --checkpoint runs/ppo --device cuda --fp16-encoder \
  --stop-head runs/imitation/stop_head.pt \
  --seconds 40 --max-frame-age-ms 500
```

An explicit checkpoint path loads exactly that policy. A directory loads its
newest numbered checkpoint.
