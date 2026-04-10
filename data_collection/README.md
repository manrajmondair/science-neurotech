# Data Collection

All recordings from the SciFi headstage (device: `funky-frisky-gerbil`, serial: `SFI100112`).

Each run folder contains:
- `metadata.json` — full channel stats, sample rate, duration, labels
- `neural_channels.png` — time-series plot of sampled neural channels
- `label_channels.png` — time-series of controller ground truth signals
- `correlation_heatmap.png` — neural↔label correlation matrix

> Raw HDF5 files are stored locally in `data/recordings/` (too large for git).

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

---

## Runs

| Run | Description | Duration | Channels | Status |
|-----|-------------|----------|----------|--------|
| [001](run_001_idle_baseline/) | Idle baseline — no controller input | 6.7s | 35 | Baseline |
| [002](run_002_active_joystick/) | Active joystick + A button input | 20.7s | 35 | Training data |
