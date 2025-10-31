# -*- coding: utf-8 -*-
"""
Centralize all user-tunable configuration and constants.
"""

import os
import torch
from pathlib import Path

# -------- Modes --------
OFFLINE_MODE: bool = True # set to False in online vehicle test
OFFLINE_IMAGE_DIR: str = "img_madison_1"
AUTO_EXIT_WHEN_DONE: bool = True

# -------- Model/Calib --------
CKPT: str = "log/ppgeo100702/best_epoch=37-val_loss_l2=0.038.ckpt"  # use different ckpt under different scenarios
CALIB_PATH: str = "data/0923set_4_can_bus_madison.json"  # use different calibration file under different camera position
DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"

# -------- Sender / dynamics --------
SEND_CONTROL_SIGNAL: bool = False # set to True in online vehicle test
DT: float = 0.3
OFFLINE_READ_INTERVAL_SEC: float = DT
FT_TO_M: float = 0.3048
FRONT_OVERHANG: float = 3.0 * FT_TO_M
SEND_EMBLEM_COORDS: bool = False

# -------- Visualization toggles --------
DRAW_CAMERA_OVERLAY: bool = True
SHOW_BEV_PANEL: bool = True
DRAW_GLOBAL_PATH: bool = False
SAVE_PREDICTION_IMAGES: bool = True

# -------- Ground grid (z=0) --------
DRAW_GROUND_GRID: bool = False
GRID_X_MAX_M: float = 30.0
GRID_Y_HALF_M: float = 3.0
GRID_DX_M: float = 1.0
GRID_DY_M: float = 0.5
GRID_SAMPLING_STEP_M: float = 0.15
GRID_ALPHA: float = 0.35
GRID_COLOR_BGR = (0, 200, 0)

# -------- Undistortion --------
USE_UNDISTORTION: bool = False

# -------- Gradient band params --------
BASE_COLOR_BGR = (237, 149, 100)
EDGE_LINE_COLOR_BGR = (225, 105, 65)

VEHICLE_BAND_WIDTH_M: float = 2.00
INNER_CLEAR_RATIO: float = 0.55
SIDE_STEPS: int = 8
ALPHA_INNER: float = 0.05
ALPHA_EDGE_MAX: float = 0.55
EDGE_LINE_THICK_PX: int = 3
EDGE_LINE_ALPHA: float = 0.90
EDGE_EASE_MODE: str = "linear"   # 'cosine' | 'smoothstep' | 'quadratic' | 'linear'

CENTER_LINE_COLOR_BGR = (164, 99, 78)
CENTER_LINE_THICK_PX: int = 2
CENTER_LINE_ALPHA: float = 0.8

# -------- Longitudinal fade --------
USE_LONGITUDINAL_FADE: bool = True
FADE_START_M: float = 4.0
FADE_END_M: float = 18.0
FADE_GAMMA: float = 1.2


# -------- ROS --------
ROSBRIDGE_HOST: str = os.environ.get("ROSBRIDGE_HOST", "localhost")
ROSBRIDGE_PORT: int = int(os.environ.get("ROSBRIDGE_PORT", "9090"))
TOPIC_NAME: str = "/path/next_point"

# -------- Capture & GUI --------
TARGET_W, TARGET_H = 1600, 900
SAVE_INTERVAL_SEC: float = 0.3

OUTPUT_X_SCALE: float = 0.2
OUTPUT_Y_SCALE: float = 0.2

Y_SMOOTHING_MODE: int = 2
Y_WEIGHTS = {
    1: [0.4, 0.5, 0.1],
    2: [0.4 * 1, 0.3 * 0.5, 0.3 * 0.33],
}

PX_PER_M: int = 15
MARGIN_PX: int = 40

# -------- Paths --------
SCRIPT_DIR: Path = Path(__file__).resolve().parent

