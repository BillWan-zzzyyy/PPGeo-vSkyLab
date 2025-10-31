# -*- coding: utf-8 -*-
"""
    Orchestrate a modular vision-driven path planning pipeline with two processes:
    (1) Data source process: reads frames either from an offline image folder or a live camera,
        depending on C.OFFLINE_MODE. The process pushes image paths into an inter-process queue.
    (2) Inference/visualization process: loads model + calibration once, performs forward prediction
        per image, draws the gradient band + centerline overlay (and optional ground grid, longitudinal
        fade, BEV panel), optionally saves the visualized output, and displays it in a window.

    In parallel, the main process creates a Tk GUI (optional, depending on C.DRAW_GLOBAL_PATH)
    and starts a background PathSender thread to (a) optionally publish control points to ROS (only
    when online mode + SEND_CONTROL_SIGNAL enabled) and (b) maintain a simple global path trace
    in the GUI.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
import tkinter as tk

# Project-wide configuration (paths, toggles, constants)
import config as C

# Offline image reader is always safe to import
from data_io import offline_image_reader_process
# Only import camera capture when needed, avoiding platform-specific code in offline mode
if not C.OFFLINE_MODE:
    from data_io import capture_process

# Model loading / single-image prediction
from model_infer import load_model_and_calib, predict_one

# Geometry helpers: grid generation/projection, and single-pass alpha blending
from geometry import generate_ground_grid_polylines, cam_project_polylines, prepare_grid_layers, alpha_blend_per_pixel

# High-level visualization: gradient band + centerline, BEV panel, legend tag
from visualize import draw_edge_gradient_band, add_tag, make_bev_panel

# GUI + optional ROS sender (ROS disabled in offline mode or when SEND_CONTROL_SIGNAL=False)
from gui_sender import PathGUI, PathSender


def prediction_process(image_queue: mp.Queue, prediction_queue: mp.Queue, pred_dir: Path, quit_event: mp.Event):
    """
    Consumer/producer process:
        1) Load the planning model and camera calibration (once).
        2) Repeatedly:
            - Pop an image path from the image_queue.
            - Run a forward pass to get predicted (x,y) waypoints in the ego frame.
            - Draw gradient band + center solid line (+ optional ground grid and longitudinal fade).
            - Add a BEV panel on the right if enabled.
            - Save and show the composed visualization frame.
            - Push the raw predicted array to the prediction_queue for the sender/GUI.
        3) On exit or 'q' key press, notify the sender via a sentinel and clean up the window.

    Args:
        image_queue:  Inter-process queue of image paths produced by the data source process.
        prediction_queue: Inter-process queue to send predicted waypoints to PathSender.
        pred_dir:     Output directory for saving composed visualization frames.
        quit_event:   Shared event flag to coordinate process/thread shutdown.

    Notes:
        - Ground grid layers are prepared lazily on the first frame to amortize the cost.
        - K_for_draw is a display-time intrinsic adjusted for the training crop; we keep this
          distinction from the raw K to avoid surprising projection offsets.
        - Alpha blending is done in a single pass for performance.
    """
    print(f"[infer] saving results to: {pred_dir}")
    print("[infer] loading model/calibration...")
    try:
        # Load once to amortize model/camera setup across all frames
        model, calib_data, K_for_draw, K_raw, T_ego_to_camera = load_model_and_calib()
        print("[infer] ✅ model/calibration ready")
    except Exception as e:
        # Abort early if model or calibration is missing/invalid
        print(f"[infer] ❌ init failed: {e}"); quit_event.set(); return

    # Pre-generate ground grid geometry in ego frame (z=0); projected on first frame
    grid_polylines_ego = generate_ground_grid_polylines(
        x_max=C.GRID_X_MAX_M, y_half=C.GRID_Y_HALF_M, dx=C.GRID_DX_M, dy=C.GRID_DY_M, sample_step=C.GRID_SAMPLING_STEP_M
    )
    grid_layers_ready = False
    grid_color_layer = grid_alpha_layer = None

    window_name = "2. Prediction View"; positioned = False

    try:
        while not quit_event.is_set():
            # --- Dequeue next image path (non-blocking timeout keeps responsiveness) ---
            try:
                image_path_str = image_queue.get(timeout=1)
            except mp.queues.Empty:
                continue
            if image_path_str is None:  # Upstream has finished; exit gracefully
                break
            image_path = Path(image_path_str)
            if not image_path.exists():
                # Skip silently if file disappeared between enqueue and now
                continue

            # --- Run model prediction for this image ---
            pred_xy = predict_one(model, str(image_path))
            # Forward prediction to the sender (GUI/ROS)
            try:
                prediction_queue.put_nowait(pred_xy)
            except mp.queues.Full:
                # Sender is temporarily behind; drop this frame's pred to keep UI smooth
                print("[infer] ⚠ send queue full, drop pred")

            # --- Load the image for visualization (undistortion is optional) ---
            raw_img = cv2.imread(str(image_path))
            display_img = raw_img  # undistort optional
            K_draw = K_for_draw
            h, w, _ = display_img.shape

            # --- Prepare and blend the ground grid (first frame only) ---
            if C.DRAW_GROUND_GRID and not grid_layers_ready:
                grid_uv_list = cam_project_polylines(grid_polylines_ego, T_ego_to_camera, K_draw)
                grid_color_layer, grid_alpha_layer = prepare_grid_layers(
                    h, w, grid_uv_list, color_bgr=C.GRID_COLOR_BGR, alpha=C.GRID_ALPHA, thickness=1
                )
                grid_layers_ready = True

            if C.DRAW_GROUND_GRID and grid_layers_ready:
                display_img = alpha_blend_per_pixel(display_img, grid_color_layer, grid_alpha_layer)

            # --- Draw the trajectory gradient band + center solid line (+ optional longitudinal fade) ---
            if C.DRAW_CAMERA_OVERLAY:
                display_img = draw_edge_gradient_band(
                    display_img, pred_xy, T_ego_to_camera, K_draw,
                    total_width_m=C.VEHICLE_BAND_WIDTH_M,
                    inner_clear_ratio=C.INNER_CLEAR_RATIO,
                    side_steps=C.SIDE_STEPS,
                    alpha_inner=C.ALPHA_INNER,
                    alpha_edge_max=C.ALPHA_EDGE_MAX,
                    edge_line_alpha=C.EDGE_LINE_ALPHA,
                    edge_thick_px=C.EDGE_LINE_THICK_PX,
                    base_color_bgr=C.BASE_COLOR_BGR,
                    edge_color_bgr=C.EDGE_LINE_COLOR_BGR,
                    ease_mode=C.EDGE_EASE_MODE,
                    center_color_bgr=C.CENTER_LINE_COLOR_BGR,
                    center_thick_px=C.CENTER_LINE_THICK_PX,
                    center_alpha=C.CENTER_LINE_ALPHA,
                    use_longitudinal_fade=C.USE_LONGITUDINAL_FADE,
                    fade_start_m=C.FADE_START_M,
                    fade_end_m=C.FADE_END_M,
                    fade_gamma=C.FADE_GAMMA
                )

            # --- Add legend tag and optional BEV panel on the right ---
            add_tag(display_img, image_path.name, show_grid=C.DRAW_GROUND_GRID, base_color_bgr=C.BASE_COLOR_BGR, grid_color_bgr=C.GRID_COLOR_BGR)

            if C.SHOW_BEV_PANEL:
                right_img = make_bev_panel(pred_xy, height=h, width=w)
                combined_img = np.concatenate([display_img, right_img], axis=1)
            else:
                combined_img = display_img

            # --- Show and optionally save the composed visualization ---
            cv2.imshow(window_name, combined_img)
            if not positioned:
                cv2.moveWindow(window_name, 0, 0); positioned = True

            if C.SAVE_PREDICTION_IMAGES:
                out = pred_dir / f"pred_{image_path.name}"
                cv2.imwrite(str(out), combined_img)

            # Press 'q' to request a clean shutdown across processes/threads
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[infer] 'q' pressed"); quit_event.set(); break
    except KeyboardInterrupt:
        # Allow Ctrl+C to break the loop gracefully
        pass
    finally:
        # Ensure window resources are released and notify the sender to stop
        cv2.destroyAllWindows()
        print("[infer] stopped.")
        prediction_queue.put(None)


if __name__ == "__main__":
    # Use 'spawn' on CUDA setups to avoid forking CUDA contexts into child processes
    if C.DEVICE.startswith("cuda"):
        mp.set_start_method('spawn', force=True)

    # Shared queues/events across processes
    image_queue, prediction_queue, quit_event = mp.Queue(maxsize=10), mp.Queue(maxsize=10), mp.Event()

    # Output directory for composed frames
    pred_results_dir = C.SCRIPT_DIR / f"pred_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pred_results_dir.mkdir(parents=True, exist_ok=True)

    # Human-readable boot log
    print("="*56)
    print("Vision-driven ROS Path Planning & Control (v7.12, modular)")
    print("="*56)
    print(f"Mode: {'Offline' if C.OFFLINE_MODE else 'Live Capture'}")
    print(f"Output dir: {pred_results_dir}")

    # Select the producer process based on OFFLINE_MODE
    if C.OFFLINE_MODE:
        # Offline: stream filenames from a folder at fixed rate
        offline_dir = Path(C.OFFLINE_IMAGE_DIR)
        process_to_run, process_args = offline_image_reader_process, (image_queue, offline_dir, quit_event)
    else:
        # Online: capture frames from a camera and save them; push paths to queue
        captures_dir = C.SCRIPT_DIR / f"captures_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        captures_dir.mkdir(parents=True, exist_ok=True)
        process_to_run, process_args = capture_process, (image_queue, captures_dir, quit_event)

    # Launch producer (data source) and consumer (inference) processes
    p_data_source = mp.Process(target=process_to_run, args=process_args)
    p_predict = mp.Process(target=prediction_process, args=(image_queue, prediction_queue, pred_results_dir, quit_event))
    p_data_source.start(); p_predict.start()

    # Tk GUI + sender thread (ROS disabled automatically in offline mode or when SEND_CONTROL_SIGNAL=False)
    sender = gui = None
    try:
        root = tk.Tk()
        gui = PathGUI(root, quit_event)
        if not C.DRAW_GLOBAL_PATH:
            # Hide the window if global path display is not requested
            root.withdraw()

        sender = PathSender(prediction_queue, quit_event, is_offline=C.OFFLINE_MODE, gui=gui)
        sender.start()

        # Periodically poll for the quit signal to close the Tk loop cleanly
        def check_for_quit():
            if quit_event.is_set():
                root.destroy()
            else:
                root.after(250, check_for_quit)

        root.after(250, check_for_quit)
        print("🚀 started. press 'q'/close window/Ctrl+C to exit.")
        root.mainloop()
    except KeyboardInterrupt:
        # Graceful termination on Ctrl+C
        print("\n[main] Ctrl+C detected")
    except tk.TclError:
        # Tk window may be closed externally (e.g., by user)
        print("\n[main] GUI closed")
    finally:
        # Ensure all workers are signaled and joined
        if not quit_event.is_set():
            quit_event.set()
        if sender:
            sender.join(timeout=2)
        p_data_source.join(timeout=2)
        p_predict.join(timeout=2)
        print("\n🏁 all done. exiting.")
