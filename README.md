# Weapon-detection-model

Webcam-based surveillance prototype that uses YOLO models to detect people, detect weapons, estimate threat level, and save evidence snapshots.

## Clone and run

This repository stores model weights with Git LFS. Install Git LFS before cloning or pull the LFS files after cloning.

```bash
git lfs install
git clone https://github.com/Saurya-0301/Weapon-detection-model.git
cd Weapon-detection-model
git lfs pull
```

## Main files

- `surveillance_ai.py`: OpenCV surveillance app with person detection, weapon detection, tracking, and evidence capture
- `dashboard.py`: Streamlit dashboard view
- `train_weapon.py`: training entry point for the custom weapon detector
- `models/`: shared model weights used by the app

## Required shared model files

- `models/yolov8x.pt`
- `models/yolov8m.pt`
- `models/yolov8s.pt`
- `models/weapon_best.pt`
