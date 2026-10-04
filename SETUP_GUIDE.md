# Tether Room – Setup Guide

Get Tether Room running on a new Windows PC. Takes about 10 minutes.

## 1. Prerequisites (install once)

| Tool | Version | Check in a terminal | Download |
|---|---|---|---|
| Python | 3.10 or newer | `python --version` | https://www.python.org/downloads/ |
| Git | any recent | `git --version` | https://git-scm.com/download/win |

- When installing Python, **tick "Add python.exe to PATH"**.
- Optional: an NVIDIA GPU makes cropping about 3x faster (see step 4).

## 2. Get the code

Open PowerShell or Command Prompt and run:

```
git clone https://github.com/MFFOTO/runner-tether-room.git
cd runner-tether-room
```

To update later: `git pull`

## 3. Install everything

```
setup.bat
```

This script does all of the following automatically:

1. Creates a private Python environment in `.venv`
2. Installs all libraries from `requirements.txt` (opencv, ultralytics/torch, numpy, piexif, tqdm, psutil)
3. Downloads the crop engine (`runner_suite_core.py`) from `MFFOTO/runner-suite-stage6`
4. Downloads the colour LUTs (`luts/*.cube`) from `MFFOTO/runner-lut-suite`
5. Creates your personal `settings_merged.json`

If it ends with `Done.` everything is in place. If you see `[ERROR]`, read the message – it is usually a missing Python on PATH or no internet connection.

## 4. (Optional) NVIDIA GPU

Only if the PC has an NVIDIA graphics card:

```
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Check that it works:

```
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

`True` = GPU is used. `False` = CPU only (still works, just slower).

## 5. Check your settings

Open `settings_merged.json` and adjust the paths for this PC. The important one:

- `merged.note_dir` – where the delivery note is written. The default is `D:/ProLoad Bulk Loader/Release`. If that folder does not exist on this PC, set it to `""` (the note is then saved next to the photos).

This file is personal to each machine and is not overwritten by `git pull`.

## 6. Start

```
run.bat
```

The browser opens `http://127.0.0.1:8770/`. Choose the folder your tethering software writes into – cropping starts right away.

On the **first run**, the YOLO model (`yolov8m-pose.pt`, ~50 MB) is downloaded automatically, so internet is needed once.

## Updating later

```
git pull
.venv\Scripts\python.exe fetch_deps.py --update
```

The second line pulls the newest crop engine and LUTs.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Python was not found on PATH` | Reinstall Python and tick "Add to PATH", then reopen the terminal |
| `Not set up yet` when running `run.bat` | Run `setup.bat` first |
| `The crop engine is missing` | `.venv\Scripts\python.exe fetch_deps.py` |
| `git` not recognised | Install Git, reopen the terminal |
| Very slow cropping | No GPU torch – see step 4 |
