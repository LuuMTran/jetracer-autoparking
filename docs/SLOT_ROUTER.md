# Three-frame policy router

The route classifier reuses the MobileNetV2 encoder from a trained direct-parking
imitation policy. It independently encodes the first three frames, averages the
three 1280-dimensional embeddings, and applies a `128 -> 1` binary head.

The positive label means that the empty slot is on the left. At runtime:

- probability at or above the saved threshold: run direct then reverse;
- probability below the threshold: run direct only.

The car remains neutral while these frames are captured.

## Label file

`train_slot_classifier.py` expects a CSV with these columns:

```text
episode,label,frames
episode_001,1,"[\"dataset/episode_001/frames/000000.jpg\", \"dataset/episode_001/frames/000001.jpg\", \"dataset/episode_001/frames/000002.jpg\"]"
```

The dataset argument is the ZIP archive containing the referenced paths. Use
`1` for left and `0` for middle or right. Split assignment is stratified by
episode, and sequence-consistent augmentation is applied only to training data.

```bash
python3 train_slot_classifier.py \
  --dataset direct_dataset.zip \
  --labels slot_labels.csv \
  --policy-checkpoint runs/direct_imitation/best.pt \
  --output runs/slot_classifier \
  --epochs 20 --finetune-epochs 8 --device cuda
```

The output `slot_left_classifier.pt` contains the model, threshold, class names,
and held-out metrics required by `SlotPredictor`.
