# -*- coding: utf-8 -*-
"""
Misc utilities (device discovery, timers, safe ops).
"""
from __future__ import annotations
from typing import Optional
from pathlib import Path
import subprocess
import re
import cv2

def find_action5_device() -> Optional[str]:

    """Discover a UVC device likely to be DJI Osmo Action and return its /dev/videoX path."""
    
    candidates: list[str] = []
    byid = Path("/dev/v4l/by-id")
    if byid.exists():
        for p in byid.iterdir():
            name = p.name.lower()
            if any(k in name for k in ["osmo", "action", "dji"]):
                try:
                    real = p.resolve()
                    if real.exists() and "video" in real.name:
                        candidates.append(str(real))
                except Exception:
                    pass
    if not candidates:
        try:
            out = subprocess.check_output(["v4l2-ctl", "--list-devices"], text=True, stderr=subprocess.STDOUT)
            blocks = re.split(r"\n\s*\n", out.strip())
            for b in blocks:
                header, *rest = b.splitlines()
                if any(k in header.lower() for k in ["osmo", "action", "dji"]):
                    for line in rest:
                        line = line.strip()
                        if line.startswith("/dev/video"):
                            candidates.append(line)
        except Exception:
            pass
    if not candidates:
        candidates.extend([str(p) for p in sorted(Path("/dev").glob("video*"))])

    for dev in candidates:
        cap = cv2.VideoCapture(dev)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            return dev
    return None
