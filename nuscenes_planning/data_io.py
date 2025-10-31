# -*- coding: utf-8 -*-
"""
Image acquisition (offline/online) and preprocessing.
"""

from __future__ import annotations
from pathlib import Path
import time
import multiprocessing as mp
import cv2
from datetime import datetime
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import config as C

def preprocess_image(img_path: str) -> "torch.Tensor":

    """
    Load and crop/resize to match training input (ImageNet mean/std normalization).
    Returns a 3xHxW tensor.
    Example:
        t = preprocess_image('/path/to/img.jpg')  # ready for model.forward(t[None])
    """

    from utils import find_action5_device
    ORIG_W, ORIG_H = 1600, 900
    FINAL_H, FINAL_W = 224, 480
    SCALE = 0.3
    TOP_CROP = 46
    img = Image.open(img_path).convert("RGB")
    if img.size != (ORIG_W, ORIG_H):
        img = img.resize((ORIG_W, ORIG_H), resample=Image.BILINEAR)
    resize_dims = (int(ORIG_W * SCALE), int(ORIG_H * SCALE))
    img = img.resize(resize_dims, resample=Image.BILINEAR)
    crop_w = int(max(0, (resize_dims[0] - FINAL_W) / 2))
    left, top = crop_w, TOP_CROP
    right, bottom = crop_w + FINAL_W, TOP_CROP + FINAL_H
    img = img.crop((left, top, right, bottom))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize(to_tensor(img))

def offline_image_reader_process(image_queue: mp.Queue, image_dir: Path, quit_event: mp.Event):

    """Producer process: push image path strings at fixed interval."""

    print(f"[offline] reading from: {image_dir}")
    if not image_dir.exists() or not image_dir.is_dir():
        print(f"[offline] ❌ folder not found: {image_dir}")
        quit_event.set(); image_queue.put(None); return
    image_paths = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    if not image_paths:
        print(f"[offline] ❌ no images found in {image_dir}")
        quit_event.set(); image_queue.put(None); return
    print(f"[offline] ✅ found {len(image_paths)} images @ {1.0/C.OFFLINE_READ_INTERVAL_SEC:.1f} Hz")
    try:
        for fpath in image_paths:
            if quit_event.is_set(): break
            try:
                image_queue.put(str(fpath), timeout=2)
            except mp.queues.Full:
                print("[offline] ⚠ queue full, waiting...")
                continue
            time.sleep(C.OFFLINE_READ_INTERVAL_SEC)
    except KeyboardInterrupt:
        pass
    finally:
        print("[offline] stopped."); image_queue.put(None)

def capture_process(image_queue: mp.Queue, out_dir: Path, quit_event: mp.Event):

    """Producer process: capture camera frames and save to disk, push their paths."""
    
    print(f"[capture] saving to: {out_dir}")
    device = find_action5_device()
    if not device:
        print("[capture] ❌ no camera found."); quit_event.set(); return
    print(f"[capture] 🎥 using: {device}")
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"[capture] ❌ cannot open {device}."); quit_event.set(); return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.TARGET_H)
    window_name = "1. Raw Camera Capture"
    positioned = False
    last_save = 0.0

    try:
        while not quit_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.1); continue
            if frame.shape[1] != C.TARGET_W or frame.shape[0] != C.TARGET_H:
                frame = cv2.resize(frame, (C.TARGET_W, C.TARGET_H))
            frame = cv2.flip(frame, -1)  # adjust if needed

            cv2.imshow(window_name, frame)
            if not positioned:
                cv2.moveWindow(window_name, 0, 0); positioned = True

            now = time.time()
            if now - last_save >= C.SAVE_INTERVAL_SEC:
                last_save = now
                fname = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                fpath = out_dir / fname
                cv2.imwrite(str(fpath), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                try:
                    image_queue.put_nowait(str(fpath))
                except mp.queues.Full:
                    print("[capture] ⚠ queue full, drop frame")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[capture] 'q' pressed"); quit_event.set(); break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[capture] stopped.")
        image_queue.put(None)
