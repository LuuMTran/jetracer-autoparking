# Streamlit parking controller

The Streamlit page launches a hardware worker for each run. The worker classifies
three stationary frames, selects direct-only or direct-then-reverse, records the
camera, and overlays steering and throttle meters. It writes an MP4, telemetry
CSV, preview image, status JSON, and worker log below `gui_runs/`.

The worker accepts artifact paths through `PARKING_DIRECT_CHECKPOINT`,
`PARKING_DIRECT_STOP_HEAD`, `PARKING_REVERSE_CHECKPOINT`,
`PARKING_REVERSE_STOP_HEAD`, and `PARKING_SLOT_CLASSIFIER`. Their defaults use
the generic paths shown in the main README.

```bash
python3 -m pip install --user -r requirements-gui.txt

PARKING_DIRECT_CHECKPOINT=runs/direct_ppo \
PARKING_DIRECT_STOP_HEAD=runs/direct_imitation/stop_head.pt \
PARKING_REVERSE_CHECKPOINT=runs/reverse_ppo \
PARKING_REVERSE_STOP_HEAD=runs/reverse_stop/stop_head.pt \
PARKING_SLOT_CLASSIFIER=runs/slot_classifier/slot_left_classifier.pt \
python3 -m streamlit run 07_streamlit_overall.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

Open `http://JETSON_IP:8501`. The status card displays the left-slot probability
and selected route before movement begins.
