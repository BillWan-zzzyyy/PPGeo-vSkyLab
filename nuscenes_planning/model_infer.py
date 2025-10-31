# -*- coding: utf-8 -*-
"""
Load planning model + calibration; run forward; build draw-time K/T.
"""

from __future__ import annotations
import os
import json
from pathlib import Path
import numpy as np
import torch
from pyquaternion import Quaternion
from data_io import preprocess_image
import config as C

# The user provided this external class in their project:
from planning_model import Planning_Model  # noqa: F401

def load_model_and_calib() -> tuple[torch.nn.Module, dict, np.ndarray, np.ndarray, np.ndarray]:

    """
    Load model (eval mode) and calibration.
    Returns:
        model: torch model in eval mode on C.DEVICE
        calib_data: dict from json[0]
        K_for_draw: (3,3) intrinsic (cropped principal point)
        K_raw: (3,3) original intrinsic
        T_ego_to_camera: (4,4) homogeneous transform
    """

    if not os.path.exists(C.CKPT):
        raise FileNotFoundError(f"ckpt not found: {C.CKPT}")
    if not os.path.exists(C.CALIB_PATH):
        raise FileNotFoundError(f"calib not found: {C.CALIB_PATH}")

    model = Planning_Model().to(C.DEVICE).eval()
    ckpt_raw = torch.load(C.CKPT, map_location=C.DEVICE)
    state_dict = ckpt_raw.get("state_dict", ckpt_raw)
    state = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state, strict=False)
    torch.backends.cudnn.benchmark = True

    with open(C.CALIB_PATH, "r") as f:
        calib_data = json.load(f)[0]

    K = np.array(calib_data["camera_intrinsic"], dtype=float)
    D = np.array(calib_data.get("distortion_coefficients", [0,0,0,0,0]), dtype=float)
    K_adjusted = K.copy(); K_adjusted[1, 2] -= 46  # training-time crop shift

    T_sensor_in_ego = np.eye(4, dtype=float)
    T_sensor_in_ego[:3, :3] = Quaternion(calib_data["rotation"]).rotation_matrix
    T_sensor_in_ego[:3, 3]  = np.array(calib_data["translation"], dtype=float)

    R_axis_change_h = np.eye(4, dtype=float)
    R_axis_change_h[:3, :3] = np.array([[0, -1, 0],
                                        [0,  0,-1],
                                        [1,  0, 0]], dtype=float)
    T_ego_to_camera = R_axis_change_h @ np.linalg.inv(T_sensor_in_ego)
    return model, calib_data, K_adjusted, K, T_ego_to_camera

def predict_one(model: torch.nn.Module, img_path: str) -> np.ndarray:

    """
    Run a single forward pass on an image path.
    Returns:
        pred_xy: (N,2) numpy array in ego frame [x_forward, y_left] (meters)
    """
    
    with torch.no_grad():
        t = preprocess_image(img_path)
        pred = model(t.unsqueeze(0).to(C.DEVICE))
    return pred.squeeze().cpu().numpy()
