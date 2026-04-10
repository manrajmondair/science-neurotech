# Science NeuroTech — Global NeuroHack 2026

BCI decoder and game built on the Science Corporation SciFi headstage for the Global NeuroHack hackathon.

## Team
- Manraj
- Shay
- Yoyo
- Medha

## Project Overview

Maximizing bit rate through a neural decoder pipeline:
1. **Decoder**: ONNX model trained to decode controller inputs from mock neural data
2. **Synapse App**: On-device inference running on the SciFi headstage
3. **Game**: Interactive game scored by achieved bit rate over 60 seconds
4. **Hardware Integration**: Robotic arm for physical game interaction

## Architecture

```
Controller → SciFi (mock neural encoder) → Synapse App (ONNX decoder) → Game/Robotic Arm
```

## Getting Started

### Prerequisites
- Python 3.10+
- Docker (for Synapse App builds)
- `pip install --pre science-synapse`

### Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### SciFi Device Setup
```bash
# Discover device
synapsectl discover

# Check device info
synapsectl -u <device-ip> info

# Connect controller: hold B button while plugging USB
# Verify encoder peripherals appear
synapsectl -u <device-ip> info
```

### Data Collection
```bash
python scripts/collect_data.py --device <device-ip> --mode easy --duration 300
```

### Training
```bash
python scripts/train_decoder.py --data data/recordings/ --output models/decoder.onnx
```

## Project Structure
```
├── data/               # Training data and recordings
├── models/             # Trained ONNX models
├── scripts/            # Data collection, training, utilities
├── synapse_app/        # On-device Synapse App (C++)
├── game/               # Game application
└── arm/                # Robotic arm control
```

## Resources
- [SciFi Docs](https://science.xyz/docs/d/scifi1/index)
- [Synapse Docs](https://science.xyz/docs/c/synapse)
- [Synapse Apps Docs](https://science.xyz/docs/d/synapse-app/index)
- [Synapse Example App](https://github.com/sciencecorp/synapse-example-app)
