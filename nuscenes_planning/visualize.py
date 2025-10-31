# -*- coding: utf-8 -*-
"""
Visualization: High-level drawing (gradient band with centerline, BEV panel, tags).
"""

from __future__ import annotations
import numpy as np
import cv2
import math
from geometry import cam_to_uv, alpha_blend_per_pixel, make_longitudinal_fade_map

def add_tag(img: np.ndarray, fname: str, show_grid: bool, base_color_bgr=(237,149,100), grid_color_bgr=(0,200,0)) -> None:

    """Draw a semi-transparent legend box."""

    legend = ["Prediction"]
    colors = [base_color_bgr]
    if show_grid:
        legend.append("Ground Grid (z=0)")
        colors.append(grid_color_bgr)
    font, scale, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 6
    (tw, th), _ = cv2.getTextSize(fname, font, scale, 1)
    box_w = tw + pad * 2
    box_h = th + pad * 2 + len(legend) * 18
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    cv2.putText(img, fname, (10 + pad, 10 + th + pad - 2), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    y0 = 10 + th + pad + 8
    for i, (txt, col) in enumerate(zip(legend, colors)):
        y = y0 + i * 18
        cv2.rectangle(img, (10 + pad, y - 10), (10 + pad + 15, y + 5), col, -1)
        cv2.putText(img, txt, (10 + pad + 22, y + 2), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

def make_bev_panel(pred_ego_xy: np.ndarray, height: int, width: int) -> np.ndarray:

    """Render a simple top-down view (ego frame) with labeled points."""

    panel = np.full((height, width, 3), 255, np.uint8)
    margin = 30
    if pred_ego_xy.size == 0:
        return panel
    x, y = pred_ego_xy[:, 0].astype(float), pred_ego_xy[:, 1].astype(float)
    x_min, x_max = float(min(np.min(x), 0.0)), float(max(np.max(x), 1.0))
    x_rng = max(x_max - x_min, 1e-6)
    max_abs_y = float(max(np.max(np.abs(y)), 1.0))
    scale = max(1e-6, min((height - 2 * margin) / x_rng, (width - 2 * margin) / (2 * max_abs_y)))
    origin_v, origin_u = height - margin, width // 2
    pts = np.array([[int(round(origin_u - yi * scale)), int(round(origin_v - xi * scale))]
                    for xi, yi in zip(x, y)], dtype=int)

    cv2.line(panel, (origin_u, margin), (origin_u, origin_v), (220, 220, 220), 1, cv2.LINE_AA)
    for i in range(len(pts) - 1):
        cv2.line(panel, tuple(pts[i]), tuple(pts[i+1]), (0, 0, 255), 3, cv2.LINE_AA)
    for p in pts:
        cv2.circle(panel, tuple(p), 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(panel, (origin_u, origin_v), 7, (0, 128, 0), -1, cv2.LINE_AA)

    font, fscale, thick, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 3
    for i, ((u, v), xi, yi) in enumerate(zip(pts, x, y)):
        label = f"({xi:+.3f}, {yi:+.3f})"
        (tw, th), _ = cv2.getTextSize(label, font, fscale, thick)
        tx, ty = (u + 8, v - 8) if i % 2 == 0 else (u - tw - 8, v + th + 8)
        tx = max(pad, min(tx, width - tw - pad))
        ty = max(th + pad, min(ty, height - pad))
        cv2.rectangle(panel, (tx - pad, ty - th - pad), (tx + tw + pad, ty + pad), (255, 255, 255), -1)
        cv2.putText(panel, label, (tx, ty), font, fscale, (20, 20, 20), thick, cv2.LINE_AA)

    cv2.arrowedLine(panel, (width-margin-20, height-margin-12), (width-margin-20, height-margin-70), (80,80,80), 2, tipLength=0.25)
    cv2.putText(panel, "x+ (forward)", (width-margin-100, height-margin-70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80,80,80), 1)
    cv2.arrowedLine(panel, (width-margin-20, height-margin-70), (width-margin-80, height-margin-70), (80,80,80), 2, tipLength=0.25)
    cv2.putText(panel, "y+ (left)", (width-margin-100, height-margin-80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80,80,80), 1)
    cv2.putText(panel, "Top-down View (Ego Coordinates)", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (60,60,60), 2, cv2.LINE_AA)
    return panel

def draw_edge_gradient_band(
    img_bgr: np.ndarray,
    pred_xy_m: np.ndarray,           # (N,2) in ego [x_forward, y_left] (m)
    T_ego_to_camera: np.ndarray,     # (4,4) ego -> camera
    K: np.ndarray,                   # (3,3) camera intrinsics
    *,
    total_width_m: float = 2.60,
    inner_clear_ratio: float = 0.45,
    side_steps: int = 5,
    alpha_inner: float = 0.05,
    alpha_edge_max: float = 0.65,
    edge_line_alpha: float = 0.90,
    edge_thick_px: int = 3,
    base_color_bgr=(235,206,135),
    edge_color_bgr=(245,220,190),
    ease_mode: str = "cosine",
    center_color_bgr=(255,255,255),
    center_thick_px: int = 2,
    center_alpha: float = 0.85,
    use_longitudinal_fade: bool = False,
    fade_start_m: float = 5.0,
    fade_end_m: float = 35.0,
    fade_gamma: float = 1.2
) -> np.ndarray:

    """
    Draw a center-clarified, side-deepened gradient band with a solid centerline.
    Optionally apply a longitudinal fade (farther => more transparent).
    """
    
    h, w = img_bgr.shape[:2]
    if pred_xy_m is None or len(pred_xy_m) < 2:
        return img_bgr

    def _ease(u: float) -> float:
        u = float(max(0.0, min(1.0, u)))
        if ease_mode == "cosine":
            return 0.5 - 0.5 * math.cos(math.pi * u)
        if ease_mode == "smoothstep":
            return u*u*(3 - 2*u)
        if ease_mode == "quadratic":
            return u*u
        return u

    def _proj_lr(half_w_m: float):
        N = pred_xy_m.shape[0]
        left_ego  = np.stack([pred_xy_m[:,0], pred_xy_m[:,1] + half_w_m, np.zeros(N)], axis=0)
        right_ego = np.stack([pred_xy_m[:,0], pred_xy_m[:,1] - half_w_m, np.zeros(N)], axis=0)
        left_cam  = (T_ego_to_camera @ np.vstack([left_ego,  np.ones((1,N))]))[:3]
        right_cam = (T_ego_to_camera @ np.vstack([right_ego, np.ones((1,N))]))[:3]
        l_uv = cam_to_uv(left_cam,  K)
        r_uv = cam_to_uv(right_cam, K)
        l = l_uv[~np.isnan(l_uv).any(axis=1)].astype(int)
        r = r_uv[~np.isnan(r_uv).any(axis=1)].astype(int)
        return l, r

    def _proj_center():
        N = pred_xy_m.shape[0]
        center_ego = np.stack([pred_xy_m[:,0], pred_xy_m[:,1], np.zeros(N)], axis=0)
        center_cam = (T_ego_to_camera @ np.vstack([center_ego, np.ones((1,N))]))[:3]
        uv = cam_to_uv(center_cam, K)
        pts = uv[~np.isnan(uv).any(axis=1)].astype(int)
        return pts

    def _draw_solid_polyline(dst_color, dst_alpha, pts, color, alpha_val, thick):
        if pts.shape[0] < 2:
            return
        a = float(max(0.0, min(1.0, alpha_val)))
        for i in range(len(pts)-1):
            p0 = (int(pts[i,0]),   int(pts[i,1]))
            p1 = (int(pts[i+1,0]), int(pts[i+1,1]))
            cv2.line(dst_color, p0, p1, color, thick, cv2.LINE_AA)
            cv2.line(dst_alpha, p0, p1, a,     thick, cv2.LINE_AA)

    color_layer = np.zeros((h, w, 3), np.uint8)
    alpha_mask  = np.zeros((h, w),   np.float32)

    total_half  = float(total_width_m) / 2.0
    inner_half  = float(max(0.0, min(0.95, inner_clear_ratio))) * total_half
    side_steps  = max(1, int(side_steps))

    # 1) gradient shells (outer -> inner)
    for s in range(side_steps, 0, -1):
        u = s / side_steps
        e = _ease(u)
        half_w = inner_half + (total_half - inner_half) * e
        a_step = alpha_inner + (alpha_edge_max - alpha_inner) * e
        l_pts, r_pts = _proj_lr(half_w)
        if l_pts.shape[0] < 2 or r_pts.shape[0] < 2:
            continue
        poly = np.vstack([l_pts, r_pts[::-1]]).reshape((-1,1,2))
        cv2.fillPoly(color_layer, [poly], base_color_bgr)
        cv2.fillPoly(alpha_mask,  [poly], float(max(0.0, min(1.0, a_step))))

    # 2) inner core (near transparent)
    l_in, r_in = _proj_lr(inner_half)
    if l_in.shape[0] >= 2 and r_in.shape[0] >= 2:
        poly_in = np.vstack([l_in, r_in[::-1]]).reshape((-1,1,2))
        cv2.fillPoly(color_layer, [poly_in], base_color_bgr)
        cv2.fillPoly(alpha_mask,  [poly_in], float(max(0.0, min(1.0, alpha_inner))))

    # 3) outer edge lines
    l_out, r_out = _proj_lr(total_half)
    if l_out.shape[0] >= 2:
        cv2.polylines(color_layer, [l_out.reshape(-1,1,2)], False, edge_color_bgr, edge_thick_px, cv2.LINE_AA)
        cv2.polylines(alpha_mask,  [l_out.reshape(-1,1,2)], False, float(max(0.0, min(1.0, edge_line_alpha))), edge_thick_px, cv2.LINE_AA)
    if r_out.shape[0] >= 2:
        cv2.polylines(color_layer, [r_out.reshape(-1,1,2)], False, edge_color_bgr, edge_thick_px, cv2.LINE_AA)
        cv2.polylines(alpha_mask,  [r_out.reshape(-1,1,2)], False, float(max(0.0, min(1.0, edge_line_alpha))), edge_thick_px, cv2.LINE_AA)

    # 4) center solid line
    c_pts = _proj_center()
    if c_pts.shape[0] >= 2:
        _draw_solid_polyline(color_layer, alpha_mask, c_pts, center_color_bgr, center_alpha, center_thick_px)

    # 5) longitudinal fade
    if use_longitudinal_fade:
        fade_map = make_longitudinal_fade_map(h, w, T_ego_to_camera, K, fade_start_m, fade_end_m, fade_gamma)
        alpha_mask *= fade_map

    # 6) blend once
    out = alpha_blend_per_pixel(img_bgr, color_layer, alpha_mask)
    return out
