# -*- coding: utf-8 -*-
"""
Minimal Tk GUI for global path visualization and an optional ROS sender thread.
Notes:
  - ROS publishing is fully disabled in offline mode or when SEND_CONTROL_SIGNAL=False.
  - roslibpy is lazily imported only when needed (online mode + sending enabled).
  - This file uses Optional for Python 3.8/3.9 compatibility.
"""

from __future__ import annotations

import threading
import time
import math
from typing import Optional, Any

import tkinter as tk
import numpy as np

import config as C


def step_body_to_global(X: float, Y: float, Yaw: float,
                        dx_local: float, dy_local: float) -> tuple[float, float]:
    
    """Integrate a local step (ego frame) into the global frame given current yaw.

    Args:
        X, Y: Current global position.
        Yaw: Current global yaw (radians).
        dx_local, dy_local: Step in ego frame (meters), x-forward, y-left.

    Returns:
        (X_new, Y_new): Updated global position after the local step.
    """

    c, s = math.cos(Yaw), math.sin(Yaw)
    dX = c * dx_local - s * dy_local
    dY = s * dx_local + c * dy_local
    return X + dX, Y + dY


class PathGUI:

    """Lightweight canvas for stitching and displaying the global path."""

    def __init__(self, root: tk.Tk, quit_event):
        self.root, self.quit_event = root, quit_event
        self.root.title("3. Global Path Tracker")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.w, self.h = 800, 800
        self.canvas = tk.Canvas(self.root, width=self.w, height=self.h, bg="white")
        self.canvas.pack()

        # Path state (global coordinates, meters)
        self.path_points_global = [(0.0, 0.0)]
        self.xmin, self.ymin, self.xmax, self.ymax = -1.0, -3.0, 25.0, 3.0

        # Canvas elements
        self.path_line: Optional[int] = None
        self.dot = self.canvas.create_oval(-10, -10, -10, -10, fill="red", outline="")

        self.redraw_canvas()
        self.root.bind('<KeyPress-q>', self.on_key_press)

    def on_closing(self):
        print("[GUI] closed")
        self.quit_event.set()

    def on_key_press(self, _=None):
        print("[GUI] 'q' pressed")
        self.on_closing()

    def _to_px(self, p):

        """Convert (x [m], y [m]) to canvas pixels with a simple fit."""

        x, y = p
        view_w_m, view_h_m = self.xmax - self.xmin, self.ymax - self.ymin
        if view_w_m < 1e-6 or view_h_m < 1e-6:
            return 0, 0
        px_per_m = min((self.w - 2 * C.MARGIN_PX) / view_w_m,
                       (self.h - 2 * C.MARGIN_PX) / view_h_m)
        return (x - self.xmin) * px_per_m + C.MARGIN_PX, \
               self.h - ((y - self.ymin) * px_per_m + C.MARGIN_PX)

    def redraw_canvas(self):

        """Repaint the entire canvas (when bounds change)."""

        self.canvas.delete("all")

        # Path polyline
        if len(self.path_points_global) > 1:
            flat = [c for p in self.path_points_global for c in self._to_px(p)]
            self.path_line = self.canvas.create_line(*flat, fill="#222222", width=2)

        # Origin marker
        sx, sy = self._to_px((0, 0))
        self.canvas.create_oval(sx - 4, sy - 4, sx + 4, sy + 4, fill="#00AA00", outline="")

        # Current position dot
        px, py = self._to_px(self.path_points_global[-1])
        r = 6
        self.dot = self.canvas.create_oval(px - r, py - r, px + r, py + r, fill="red", outline="")

    def update_vehicle_pose(self, new_global_point):
        
        """Append a new global point and update the canvas efficiently."""

        self.path_points_global.append(new_global_point)
        x, y = new_global_point
        resized = False

        # Expand bounds with margins to keep the dot within view
        if x > self.xmax - 5:
            self.xmax = x + 15
            resized = True
        if x < self.xmin + 5:
            self.xmin = x - 15
            resized = True
        if y > self.ymax - 2:
            self.ymax = y + 5
            resized = True
        if y < self.ymin + 2:
            self.ymin = y - 5
            resized = True

        if resized:
            self.redraw_canvas()
        else:
            # Extend polyline
            if self.path_line:
                self.canvas.coords(self.path_line,
                                   *self.canvas.coords(self.path_line),
                                   *self._to_px(new_global_point))
            # Move dot
            px, py = self._to_px(new_global_point)
            r = 6
            self.canvas.coords(self.dot, px - r, py - r, px + r, py + r)


class PathSender(threading.Thread):

    """Background thread that consumes predicted points, updates GUI,
    and optionally publishes ROS messages via roslibpy.
    """
    

    def __init__(self, prediction_queue, quit_event, is_offline: bool, gui: Optional[PathGUI] = None):
        super().__init__(daemon=True)
        self.prediction_queue, self.gui, self.quit_event = prediction_queue, gui, quit_event
        self.is_offline = is_offline

        # --- ROS enablement and lazy import ---
        self.enable_ros: bool = (not C.OFFLINE_MODE) and C.SEND_CONTROL_SIGNAL
        self._roslibpy: Optional[Any] = None
        self.ros = None
        self.pub = None

        if self.enable_ros:
            try:
                import roslibpy  # lazy import; only when needed
                self._roslibpy = roslibpy
            except ImportError:
                print("[sender] roslibpy not installed; disabling ROS publish.")
                self.enable_ros = False

        # Vehicle pose (global)
        self.X, self.Y, self.Yaw = 0.0, 0.0, 0.0

    def run(self):
        # --- ROS connection setup (if enabled) ---
        if self.enable_ros:
            try:
                self.ros = self._roslibpy.Ros(host=C.ROSBRIDGE_HOST, port=C.ROSBRIDGE_PORT)
                self.ros.run()
                print(f"[sender] connecting rosbridge {C.ROSBRIDGE_HOST}:{C.ROSBRIDGE_PORT} ...")
                while not self.ros.is_connected and not self.quit_event.is_set():
                    time.sleep(0.5)
                if self.quit_event.is_set() or not self.ros.is_connected:
                    raise ConnectionError("ROS connection interrupted/failed")
                print("[sender] ✅ connected")
                self.pub = self._roslibpy.Topic(self.ros, C.TOPIC_NAME, 'geometry_msgs/msg/Point')
            except Exception as e:
                print(f"[sender] ❌ ROS connect failed: {e}")
                self.enable_ros = False
        else:
            print("[sender] ROS disabled (offline mode or SEND_CONTROL_SIGNAL=False)")

        # Precompute smoothing weights
        weights = None
        if C.Y_SMOOTHING_MODE in C.Y_WEIGHTS:
            weights = np.array(C.Y_WEIGHTS[C.Y_SMOOTHING_MODE], dtype=float)

        step_count = 0
        while not self.quit_event.is_set():
            try:
                pred_xy = self.prediction_queue.get(timeout=1)
            except Exception:
                continue
            if pred_xy is None:
                break

            # Consume the first predicted waypoint (ego frame)
            if len(pred_xy) > 0:
                dx_local = float(pred_xy[0, 0])
                raw_dy_for_log = float(pred_xy[0, 1])
                if C.Y_SMOOTHING_MODE == 0 or len(pred_xy) < 3 or weights is None:
                    dy_local = raw_dy_for_log
                else:
                    dy_local = float(np.sum(pred_xy[:3, 1] * weights))
            else:
                dx_local = dy_local = raw_dy_for_log = 0.0

            # Output scaling + emblem offset (optional)
            dx_scaled, dy_scaled = dx_local * C.OUTPUT_X_SCALE, dy_local * C.OUTPUT_Y_SCALE
            x_out = dx_scaled + (C.FRONT_OVERHANG if C.SEND_EMBLEM_COORDS else 0.0)
            y_out = dy_scaled

            # Publish or simulate
            action_log = "Simulate"
            if self.enable_ros and self.pub and self.ros.is_connected:
                self.pub.publish(self._roslibpy.Message({'x': float(x_out), 'y': float(y_out), 'z': 0.0}))
                action_log = "Send"

            step_count += 1
            print(f"[sender #{step_count:03d}] {action_log}: x={x_out:+.3f} y={y_out:+.3f} "
                  f"(raw dx={dx_local:+.3f} dy={raw_dy_for_log:+.3f})")

            # Integrate simple global pose for GUI path
            self.X, self.Y = step_body_to_global(self.X, self.Y, self.Yaw, dx_local, dy_local)
            # Yaw update via instantaneous turn (approx.)
            if abs(dx_local) > 1e-9 or abs(dy_local) > 1e-9:
                self.Yaw += math.atan2(dy_local, dx_local)

            # Update GUI in the main Tk thread
            if self.gui and C.DRAW_GLOBAL_PATH:
                self.gui.root.after(0, self.gui.update_vehicle_pose, (self.X, self.Y))

            time.sleep(C.DT)

        # Cleanup ROS if used
        if self.enable_ros and self.ros:
            try:
                if self.pub:
                    self.pub.unadvertise()
                self.ros.terminate()
            except Exception:
                pass

        print("[sender] ✅ stopped")

        # Auto shutdown when offline and everything is done
        if self.is_offline and C.AUTO_EXIT_WHEN_DONE:
            print("[sender] all images processed, requesting shutdown")
            self.quit_event.set()
