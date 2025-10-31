# -*- coding: utf-8 -*-
"""
Camera projection, grid generation, and alpha blending / longitudinal fade.
"""

from __future__ import annotations
import numpy as np
import cv2

def cam_to_uv(p_cam_3xN: np.ndarray, K: np.ndarray, eps: float = 1e-6) -> np.ndarray:

    """Project 3D points in camera coords (3xN) to image pixels using intrinsic K."""

    uv = np.full((p_cam_3xN.shape[1], 2), np.nan, dtype=float)
    valid = p_cam_3xN[2, :] > eps
    if valid.any():
        X = (K @ p_cam_3xN[:, valid]).T
        uv[valid] = (X[:, :2] / X[:, 2:3]).astype(float)
    return uv

def generate_ground_grid_polylines(x_max: float, y_half: float, dx: float, dy: float, sample_step: float) -> list[np.ndarray]:

    """Generate polylines on z=0 plane (ego frame). Each polyline is (N,3) in meters."""

    polylines: list[np.ndarray] = []
    y_vals = np.arange(-y_half, y_half + 1e-6, dy)
    xs = np.arange(0.0, x_max + 1e-6, sample_step)
    for yv in y_vals:
        pts = np.stack([xs, np.full_like(xs, yv), np.zeros_like(xs)], axis=1)
        polylines.append(pts)
    x_vals = np.arange(0.0, x_max + 1e-6, dx)
    ys = np.arange(-y_half, y_half + 1e-6, sample_step)
    for xv in x_vals:
        pts = np.stack([np.full_like(ys, xv), ys, np.zeros_like(ys)], axis=1)
        polylines.append(pts)
    return polylines

def cam_project_polylines(polylines_ego: list[np.ndarray], T_ego_to_camera: np.ndarray, K: np.ndarray) -> list[np.ndarray]:

    """Project ego-frame polylines to image UV list."""

    uv_list: list[np.ndarray] = []
    for pts in polylines_ego:
        pts_h = np.concatenate([pts.T, np.ones((1, pts.shape[0]))], axis=0)  # (4,N)
        cam = (T_ego_to_camera @ pts_h)[:3, :]
        uv = cam_to_uv(cam, K)
        uv_list.append(uv)
    return uv_list

def prepare_grid_layers(h: int, w: int, grid_uv_list: list[np.ndarray], color_bgr=(0, 200, 0), alpha: float = 0.35, thickness: int = 1):

    """Rasterize grid polylines into color and alpha layers for single-pass blending."""

    grid_color = np.zeros((h, w, 3), dtype=np.uint8)
    grid_alpha = np.zeros((h, w), dtype=np.float32)
    a = float(np.clip(alpha, 0.0, 1.0))
    for uv in grid_uv_list:
        valid = ~np.isnan(uv).any(axis=1)
        pts = uv[valid].astype(int)
        if pts.shape[0] < 2:
            continue
        for i in range(len(pts) - 1):
            cv2.line(grid_color, tuple(pts[i]), tuple(pts[i+1]), color_bgr, thickness, cv2.LINE_AA)
            cv2.line(grid_alpha, tuple(pts[i]), tuple(pts[i+1]), a,        thickness, cv2.LINE_AA)
    return grid_color, grid_alpha

def alpha_blend_per_pixel(base_img_bgr: np.ndarray, overlay_bgr: np.ndarray | None, alpha_mask_float: np.ndarray | None) -> np.ndarray:

    """Single-pass per-pixel alpha blend: out = overlay*alpha + base*(1-alpha)."""

    if overlay_bgr is None or alpha_mask_float is None:
        return base_img_bgr
    if alpha_mask_float.max() <= 0.0:
        return base_img_bgr
    base = base_img_bgr.astype(np.float32)
    over = overlay_bgr.astype(np.float32)
    a   = np.clip(alpha_mask_float, 0.0, 1.0)[:, :, None]
    out = over * a + base * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)

# -------- Longitudinal fade (z=0 ray-plane hit) --------
def _camera_to_ego(T_ego_to_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    """Return R, t for camera->ego from ego->camera."""
    
    T = np.linalg.inv(T_ego_to_camera)
    return T[:3, :3].copy(), T[:3, 3].copy()

def make_longitudinal_fade_map(h: int, w: int, T_ego_to_camera: np.ndarray, K: np.ndarray, x0: float, x1: float, gamma: float) -> np.ndarray:
    """
    Estimate forward distance x (ego) for each pixel by intersecting camera rays with z=0 plane.
    Map to fade in [0,1]: x<=x0 => 1, x>=x1 => 0, between => (1 - (x-x0)/(x1-x0))**gamma.
    """
    R_ce, t_ce = _camera_to_ego(T_ego_to_camera)
    K_inv = np.linalg.inv(K)

    u = np.arange(w, dtype=np.float64)
    v = np.arange(h, dtype=np.float64)
    U, V = np.meshgrid(u, v)
    pix = np.stack([U, V, np.ones_like(U)], axis=-1).reshape(-1, 3)
    rays_cam = (K_inv @ pix.T).T               # (N,3)
    rays_ego = (R_ce @ rays_cam.T).T           # (N,3)
    o = t_ce                                   # (3,)

    dz = rays_ego[:, 2]
    valid = (np.abs(dz) > 1e-9)
    t_hit = np.empty_like(dz)
    t_hit[:] = np.nan
    t_hit[valid] = -o[2] / dz[valid]
    valid = valid & (t_hit > 0)

    P = o[None, :] + t_hit[:, None] * rays_ego  # (N,3)
    X = P[:, 0]
    X[~valid] = np.nan

    fade = np.ones_like(X)
    span = max(1e-6, (x1 - x0))
    m = (X > x0) & (X < x1)
    fade[m] = (1.0 - ((X[m] - x0) / span)) ** gamma
    fade[X >= x1] = 0.0
    fade[np.isnan(X)] = 1.0

    return fade.reshape(h, w).astype(np.float32)
