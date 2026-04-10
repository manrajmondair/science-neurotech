# Data Collection

All recordings from the SciFi headstage (device: `funky-frisky-gerbil`, serial: `SFI100112`).

Each run folder contains:
- `metadata.json` — full channel stats, sample rate, duration, labels
- `neural_channels.png` — time-series plot of sampled neural channels
- `label_channels.png` — time-series of controller ground truth signals
- `correlation_heatmap.png` — neural↔label correlation matrix

> Raw HDF5 files stored in `data/recordings/` via **Git LFS** — teammates run `git lfs install` then `git pull`.

---

## Device Info

| Field | Value |
|-------|-------|
| Device Name | funky-frisky-gerbil |
| Serial | SFI100112 |
| IP | 192.168.16.219 |
| Sample Rate | 32,000 Hz |
| Bit Width | 12-bit |

## Peripheral IDs

| Mode | Training | Testing |
|------|----------|---------|
| Easy | 104 | 103 |
| Medium | 106 | 105 |
| Hard | 108 | 107 |

## Channel Layout (Easy Mode — Peripheral 104)

| Channels | Content | Range |
|----------|---------|-------|
| 0–31 | Encoded neural data | ~[-350, +120] raw ADC |
| 32 | Joystick X (left stick) | [-32767, +32767] |
| 33 | Joystick Y (left stick) | [-32767, +32767] |
| 34 | A Button | 0 (off) / 32767 (on) |

## Channel Layout (Hard Mode — Peripheral 108)

| Channels | Content | Range |
|----------|---------|-------|
| 0–63 | Encoded neural data (64ch) | ~[-790, +165] raw ADC |
| 64–67 | 4 analog axes (joysticks/triggers) | [-32767, +32767] |
| 68–75 | 8 binary buttons (ABXY, bumpers, etc) | 0 / 32767 |

> Exact input↔channel mapping TBD — requires isolated input testing (Run 004).

---

## Runs

| Run | Description | Duration | Channels | Mode | Status |
|-----|-------------|----------|----------|------|--------|
| [001](run_001_idle_baseline/) | Idle baseline — no controller input | 6.7s | 35 | Easy | Baseline |
| [002](run_002_active_joystick/) | Active joystick + A button input | 20.7s | 35 | Easy | Training data |
| [003](run_003_hard_mode_all_inputs/) | All inputs active (unstructured) | 3.7s | 76 | Hard | Input discovery |
| [004](run_004_structured_mapping/) | One input at a time (5s each) | 27.2s | 76 | Hard | Partial mapping |
