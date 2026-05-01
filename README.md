# Weapon-detection-model

Webcam-based surveillance prototype that uses YOLO models to detect people, detect weapons, estimate threat level, and save evidence snapshots.

## What this project includes

- `surveillance_ai.py`: main OpenCV surveillance app with person detection, weapon detection, tracking, threat scoring, and evidence capture
- `dashboard.py`: Streamlit dashboard for live monitoring
- `detect.py`: basic webcam detection using the general YOLO model
- `test_weapon_camera.py`: webcam test for the custom weapon model
- `weapon_camera_detection.py`: alternate webcam test for the custom weapon model
- `train_weapon.py`: training entry point for the custom weapon detector
- `models/`: shared model weights used by the app

## Prerequisites

- Python 3.10 or newer
- Git
- Git LFS
- A working webcam

## Clone the repository

This repository stores model weights with Git LFS. Install Git LFS before cloning or pull the LFS files after cloning.

```bash
git lfs install
git clone https://github.com/Saurya-0301/Weapon-detection-model.git
cd Weapon-detection-model
git lfs pull
```

## Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install dependencies

There is no `requirements.txt` in this repository yet, so install the packages used by the source files directly:

```bash
pip install ultralytics opencv-python torch torchvision streamlit deep-sort-realtime
```

If `torch` installation fails on your machine, install PyTorch first from the official PyTorch instructions for your OS and CUDA version, then rerun the remaining package installs.

## Model files required after clone

These files should exist after `git lfs pull`:

- `models/yolov8x.pt`
- `models/yolov8m.pt`
- `models/yolov8s.pt`
- `models/weapon_best.pt`

## Run commands

Run the main surveillance application:

```bash
python surveillance_ai.py
```

Run the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

Run the basic general-model detector:

```bash
python detect.py
```

Run the custom weapon model webcam test:

```bash
python test_weapon_camera.py
```

Run the alternate custom weapon detector script:

```bash
python weapon_camera_detection.py
```

Train the custom weapon model:

```bash
python train_weapon.py
```

## Notes for another computer

- Clone the repository normally with Git.
- Install Git LFS and run `git lfs pull` so the actual `.pt` model files download.
- Activate the virtual environment before running any command.
- Make sure the webcam is available at camera index `0`, because the current scripts use `cv2.VideoCapture(0)`.
- The repository does not include the training dataset, `runs/`, `evidence/`, or `venv/`.

## Troubleshooting

- If the app opens but models do not load, run `git lfs pull` again and verify the files exist in `models/`.
- If the webcam does not open, close other apps using the camera and try again.
- If `streamlit` is not recognized, activate the virtual environment first or run:

```bash
python -m streamlit run dashboard.py
```
