# JetRacer autonomous parking

Camera-based parking for the Waveshare/NVIDIA JetRacer. The project combines
MobileNetV2 + GRU behavior cloning, recurrent PPO with human-scored terminal
rewards, learned stop heads, a three-frame parking-slot classifier, and a
Streamlit driving/recording interface.

Model checkpoints, datasets, recordings, and generated reports are intentionally
excluded from this repository.

## Policy

```text
RGB frame -> MobileNetV2 (1280) -> projection (256)
          + previous steering/throttle (2)
          -> GRU (64)
          -> actor (32 -> steering, throttle)
          -> critic (32 -> value)
          -> stop head (32 -> stop probability)
```

Behavior cloning trains the temporal policy on 16-frame windows. PPO keeps the
encoder, projection, and GRU frozen and updates the actor, critic, and learned
exploration standard deviation. The default physical loop updates after five
human-scored episodes.

Before the overall policy moves the car, the route classifier reads three
stationary frames:

- left slot: direct policy, then reverse policy;
- middle or right slot: direct policy only.

## Repository layout

| Path | Purpose |
|---|---|
| `01_maneuvering.py` | Manual calibration and motor control |
| `03_collect_imitation.py` | Record and review demonstrations |
| `04_maneuvering_continuous.py` | Record multiple demonstrations per session |
| `02_neural_driving.py` | Train, evaluate, benchmark, and run imitation policies |
| `train_real_dataset.py` | Reproducible training with dataset audit and plots |
| `train_stop_head.py` | Train the initial terminal stop classifier |
| `05_online_ppo.py` | Initialize, collect, update, inspect, and follow PPO |
| `06_stop_tuning.py` | Collect stop labels, fine-tune stop heads, and run the conditional sequence |
| `train_slot_classifier.py` | Train the three-frame left-slot classifier |
| `07_streamlit_overall.py` | Browser control room |
| `07_streamlit_worker.py` | Hardware worker used by Streamlit |
| `08_evaluate_policy.py` | Deterministic, human-rated checkpoint evaluation |
| `09_follow_policy_offset.py` | Follow PPO with runtime steering offset and throttle scaling |
| `10_dagger.py` | Collect expert labels on policy states and fine-tune the actor with DAgger |
| `parking_lib/` | Shared camera, hardware, learning, PPO, and routing implementation |

## Installation

Collection uses the JetPack-compatible `jetracer`, `jetcam`, OpenCV, PyTorch,
and torchvision packages already installed on the Jetson image. Do not replace
the Jetson CUDA build with a generic PyPI build.

```bash
python3 -m pip install -r requirements-collection.txt
```

For desktop training:

```bash
python3 -m pip install -r requirements-training.txt
```

For the Jetson GUI running Python 3.6:

```bash
python3 -m pip install --user -r requirements-gui.txt
```

## Configuration

Edit `config.json`, verify the installed driver, steering direction, gains, and
motor limits with the wheels raised, then set `calibration_confirmed` to `true`.
Training checkpoints embed this configuration; deployment uses the checkpoint's
copy.

```bash
python3 01_maneuvering.py --real --steering-exercise
python3 01_maneuvering.py --real --controller keyboard --seconds 20
```

## Core workflow

Collect demonstrations:

```bash
python3 04_maneuvering_continuous.py \
  --real --dataset dataset --seconds 60 --scenario normal

python3 03_collect_imitation.py validate dataset
python3 03_collect_imitation.py review dataset/episode_ID \
  --outcome success --accept --score 1
```

Train MobileNetV2 + GRU:

```bash
python3 02_neural_driving.py train \
  --dataset dataset \
  --output runs/direct_imitation \
  --architecture mobilenet-v2-proj256-gru64-mlp32 \
  --augment --epochs 20 --finetune-epochs 5 --device cuda
```

Initialize PPO and run the five-episode physical update loop:

```bash
python3 05_online_ppo.py init \
  --imitation-checkpoint runs/direct_imitation/best.pt \
  --config ppo_config.json \
  --output runs/direct_ppo \
  --device cpu

JETRACER_CAPTURE_WIDTH=1920 \
JETRACER_CAPTURE_HEIGHT=1080 \
JETRACER_CAPTURE_FPS=30 \
python3 05_online_ppo.py physical-loop \
  --checkpoint runs/direct_ppo \
  --run-dir runs/direct_ppo \
  --device cuda --fp16-encoder \
  --stop-head runs/direct_imitation/stop_head.pt \
  --episodes-per-update 5 --max-frame-age-ms 500
```

Run the conditional two-policy sequence:

```bash
JETRACER_CAPTURE_WIDTH=1920 \
JETRACER_CAPTURE_HEIGHT=1080 \
JETRACER_CAPTURE_FPS=30 \
python3 06_stop_tuning.py sequence \
  --first-checkpoint runs/direct_ppo \
  --first-stop-head runs/direct_imitation/stop_head.pt \
  --second-checkpoint runs/reverse_ppo \
  --second-stop-head runs/reverse_stop/stop_head.pt \
  --slot-classifier runs/slot_classifier/slot_left_classifier.pt \
  --device cuda --fp16-encoder \
  --max-frame-age-ms 500 --second-throttle-scale 0.6
```

The terminal prints the left-slot probability and selected route before either
driving policy starts.

## DAgger correction rounds

Collect an expert-labelled episode from states visited by the current PPO
policy. Keep the controller active throughout the episode: the expert action is
saved as the training label, while `beta` controls the action sent to the car.

```bash
python3 10_dagger.py collect \
  --checkpoint runs/direct_ppo/ppo_0092.pt \
  --dataset dataset_direct \
  --controller gamepad --beta 0.5 --device cuda --fp16-encoder \
  --steering-scale 1.0 --throttle-scale 1.0

python3 10_dagger.py train \
  --checkpoint runs/direct_ppo/ppo_0092.pt \
  --dataset dataset_direct \
  --output runs/direct_dagger_round1.pt \
  --augment --epochs 10 --device cuda
```

Repeat collection and training while reducing `beta`, for example 0.5, 0.25,
then 0.0. By default only the actor changes, so the existing recurrent stop head
remains compatible. `--train-gru` adapts the temporal representation and
requires retraining the stop head.

## GUI

The worker uses generic `runs/` paths by default. Override them with the
`PARKING_DIRECT_CHECKPOINT`, `PARKING_DIRECT_STOP_HEAD`,
`PARKING_REVERSE_CHECKPOINT`, `PARKING_REVERSE_STOP_HEAD`, and
`PARKING_SLOT_CLASSIFIER` environment variables when needed, then run:

```bash
python3 -m streamlit run 07_streamlit_overall.py \
  --server.address 0.0.0.0 --server.port 8501
```

Each run saves an annotated video, telemetry CSV, live preview, status file, and
worker log below `gui_runs/`.

More details are in [`docs/PPO.md`](docs/PPO.md),
[`docs/STOP_TUNING.md`](docs/STOP_TUNING.md),
[`docs/SLOT_ROUTER.md`](docs/SLOT_ROUTER.md), and
[`docs/STREAMLIT_GUI.md`](docs/STREAMLIT_GUI.md).
