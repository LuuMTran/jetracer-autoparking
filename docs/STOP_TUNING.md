# Stop-head fine-tuning and policy sequencing

`06_stop_tuning.py` can collect human stop signals while following a frozen PPO
policy, fine-tune only the stop MLP, and run direct/reverse policies in a single
camera and motor session.

## Collect stop labels

```bash
python3 06_stop_tuning.py collect \
  --checkpoint runs/reverse_ppo \
  --output stop_labels/reverse \
  --device cuda --fp16-encoder \
  --seconds 40 --max-frame-age-ms 500
```

Enter starts a deterministic run. Press `S` at the desired stopping pose, then
press `F` after a short continuation. The collector labels the configured frame
window around `S`. A run without `S` is discarded.

## Fine-tune the stop head

```bash
python3 06_stop_tuning.py train \
  --checkpoint runs/reverse_ppo \
  --dataset stop_labels/reverse \
  --initial-stop-head runs/reverse_imitation/stop_head.pt \
  --output runs/reverse_stop \
  --device cpu --epochs 100 --patience 10
```

Only the `64 -> 32 -> 1` stop head changes. The visual encoder, projection, GRU,
actor, critic, and PPO checkpoints remain unchanged.

## Conditional sequence

```bash
python3 06_stop_tuning.py sequence \
  --first-checkpoint runs/direct_ppo \
  --first-stop-head runs/direct_imitation/stop_head.pt \
  --second-checkpoint runs/reverse_ppo \
  --second-stop-head runs/reverse_stop/stop_head.pt \
  --slot-classifier runs/slot_classifier/slot_left_classifier.pt \
  --device cuda --fp16-encoder \
  --first-seconds 40 --second-seconds 40 \
  --max-frame-age-ms 500 --second-throttle-scale 0.6
```

Three stationary frames are classified before movement. A left slot runs both
policies; a middle or right slot runs only the direct policy. `F` advances or
ends the active stage. Escape returns the motor command to neutral and exits.
