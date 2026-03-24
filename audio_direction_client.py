import json
import math
import time
import threading
from collections import deque

import numpy as np
import soundcard as sc
from websocket import create_connection
from pynput import mouse


# -----------------------------
# Hardcoded server IP / settings
# -----------------------------
WS_URL = "ws://192.168.1.50:8080/ws?role=producer"   # <-- change this
SAMPLE_RATE = 48000
BLOCK_SIZE = 1024

# Detection tuning
SHOT_MIN_GAP_MS = 180
EVENT_HOLD_MS = 850
FLUX_TRIGGER_Z = 3.5
RMS_TRIGGER_MULT = 2.4
ABS_MIN_RMS = 0.006

# Mouse tuning
MOTION_WINDOW_MS = 60
MOVE_MIN_PIXELS = 6

# UI tuning
COARSE_WIDTH_DEG = 160
MIN_REFINED_WIDTH_DEG = 24


# -----------------------------
# Shared mouse state
# -----------------------------
mouse_events = deque(maxlen=4000)
mouse_lock = threading.Lock()
_last_mouse_x = None


def on_move(x, y):
    global _last_mouse_x
    now = time.time()

    with mouse_lock:
        if _last_mouse_x is not None:
            dx = x - _last_mouse_x
            if dx != 0:
                mouse_events.append((now, dx))
        _last_mouse_x = x


def start_mouse_listener():
    listener = mouse.Listener(on_move=on_move)
    listener.daemon = True
    listener.start()
    return listener


def get_recent_mouse_dx(window_ms=MOTION_WINDOW_MS):
    cutoff = time.time() - (window_ms / 1000.0)
    total = 0.0

    with mouse_lock:
        # prune old
        while mouse_events and mouse_events[0][0] < cutoff - 1.0:
            mouse_events.popleft()

        for ts, dx in mouse_events:
            if ts >= cutoff:
                total += dx

    return total


# -----------------------------
# WebSocket publisher
# -----------------------------
class Publisher:
    def __init__(self, url):
        self.url = url
        self.ws = None
        self.last_try = 0.0

    def ensure(self):
        if self.ws is not None:
            return True

        now = time.time()
        if now - self.last_try < 2.0:
            return False

        self.last_try = now
        try:
            self.ws = create_connection(self.url, timeout=2)
            print(f"[WS] Connected to {self.url}")
            return True
        except Exception as exc:
            print(f"[WS] Connect failed: {exc}")
            self.ws = None
            return False

    def send(self, payload):
        if not self.ensure():
            return

        try:
            self.ws.send(json.dumps(payload))
        except Exception:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None


# -----------------------------
# Audio helpers
# -----------------------------
def to_stereo(block):
    arr = np.asarray(block, dtype=np.float32)

    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=1)

    if arr.shape[1] > 2:
        arr = arr[:, :2]

    return arr


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def compute_rightness(stereo):
    """
    Returns:
        rightness in [-1, 1]
        ild_db
    """
    left = stereo[:, 0]
    right = stereo[:, 1]

    l_rms = rms(left)
    r_rms = rms(right)

    ild_db = 20.0 * np.log10((r_rms + 1e-9) / (l_rms + 1e-9))

    # squash into [-1, 1]
    rightness = float(np.tanh(ild_db / 8.0))
    return rightness, float(ild_db)


def spectral_flux_mag(mono):
    windowed = mono * np.hanning(len(mono))
    mag = np.abs(np.fft.rfft(windowed))
    return mag


def open_loopback_recorder():
    """
    Best effort loopback selection.
    Works best on Windows with WASAPI loopback.
    """
    default_speaker = sc.default_speaker()
    print(f"[AUDIO] Default speaker: {default_speaker.name}")

    # Try to find matching loopback microphone
    try:
        loopbacks = sc.all_microphones(include_loopback=True)
        for mic in loopbacks:
            mic_name = getattr(mic, "name", str(mic))
            if default_speaker.name in mic_name or mic_name in default_speaker.name:
                print(f"[AUDIO] Using loopback mic: {mic_name}")
                return mic.recorder(
                    samplerate=SAMPLE_RATE,
                    channels=2,
                    blocksize=BLOCK_SIZE,
                )
    except Exception as exc:
        print(f"[AUDIO] Loopback mic search failed: {exc}")

    # Fallback
    print("[AUDIO] Falling back to speaker recorder")
    return default_speaker.recorder(
        samplerate=SAMPLE_RATE,
        channels=2,
        blocksize=BLOCK_SIZE,
    )


# -----------------------------
# Direction logic
# -----------------------------
def build_payload(event, current_rightness, current_ild_db, mouse_dx):
    side = event["side"]
    evidence = event["front_score"] + event["back_score"]
    frontish = event["front_score"] >= event["back_score"]

    # strongest observed side bias during current event
    side_bias = min(1.0, abs(event["best_rightness"]))

    if evidence < 0.8:
        # Only know left vs right hemisphere
        center = 90 if side == "right" else 270
        width = max(100, int(COARSE_WIDTH_DEG - side_bias * 35))
        confidence = min(0.55, 0.22 + side_bias * 0.33)
        mode = "coarse"
    else:
        # Front/back inference inside current side
        separation = abs(event["front_score"] - event["back_score"]) / (evidence + 1e-9)
        confidence = min(1.0, 0.50 + separation * 0.50)
        mode = "refined"

        if side == "right" and frontish:
            # front-right: 0..90, more right bias => closer to 90
            center = 15 + side_bias * 75
        elif side == "right" and not frontish:
            # back-right: 90..180, more right bias => closer to 90
            center = 165 - side_bias * 75
        elif side == "left" and frontish:
            # front-left: 360..270, more left bias => closer to 270
            center = (345 - side_bias * 75) % 360
        else:
            # back-left: 180..270, more left bias => closer to 270
            center = 195 + side_bias * 75

        width = int(max(
            MIN_REFINED_WIDTH_DEG,
            100 - confidence * 40 - min(28, evidence * 4)
        ))

    return {
        "type": "direction",
        "active": True,
        "mode": mode,
        "side": side,
        "center_deg": float(center % 360),
        "width_deg": float(width),
        "confidence": float(confidence),
        "rightness": float(current_rightness),
        "ild_db": float(current_ild_db),
        "mouse_dx": float(mouse_dx),
        "front_score": float(event["front_score"]),
        "back_score": float(event["back_score"]),
        "ts": time.time(),
    }


def main():
    start_mouse_listener()
    publisher = Publisher(WS_URL)

    prev_mag = None
    flux_mean = 1.0
    flux_dev = 1.0
    noise_rms = 0.002
    last_trigger_time = 0.0

    event = None

    print("[INFO] Starting audio direction producer...")
    print(f"[INFO] WS -> {WS_URL}")

    with open_loopback_recorder() as recorder:
        while True:
            block = recorder.record(numframes=BLOCK_SIZE)
            now = time.time()

            stereo = to_stereo(block)
            mono = stereo.mean(axis=1)

            current_rms = rms(mono)
            current_rightness, current_ild_db = compute_rightness(stereo)

            mag = spectral_flux_mag(mono)
            if prev_mag is None:
                prev_mag = mag
                continue

            flux = float(np.maximum(mag - prev_mag, 0.0).sum())
            prev_mag = mag

            # z-score-ish onset detector
            flux_z = (flux - flux_mean) / (flux_dev + 1e-6)

            # update background stats
            flux_mean = 0.98 * flux_mean + 0.02 * flux
            flux_dev = 0.98 * flux_dev + 0.02 * max(1.0, abs(flux - flux_mean))

            # update noise floor only when not inside active event
            if event is None:
                noise_rms = 0.995 * noise_rms + 0.005 * current_rms

            shot_trigger = (
                flux_z > FLUX_TRIGGER_Z
                and current_rms > max(noise_rms * RMS_TRIGGER_MULT, ABS_MIN_RMS)
                and (now - last_trigger_time) > (SHOT_MIN_GAP_MS / 1000.0)
            )

            side = "right" if current_rightness >= 0 else "left"
            mouse_dx = get_recent_mouse_dx()

            if shot_trigger:
                last_trigger_time = now

                if event is None:
                    event = {
                        "started_at": now,
                        "until": now + (EVENT_HOLD_MS / 1000.0),
                        "side": side,
                        "last_rightness": current_rightness,
                        "best_rightness": current_rightness,
                        "front_score": 0.0,
                        "back_score": 0.0,
                    }
                    print(f"[SHOT] start side={side} rightness={current_rightness:.3f}")
                else:
                    # same burst / repeated shots -> keep event alive
                    event["until"] = max(event["until"], now + (EVENT_HOLD_MS / 1000.0))

            if event is not None:
                # If side flips during event, keep original side; short blocks can wobble
                event["best_rightness"] = (
                    current_rightness
                    if abs(current_rightness) > abs(event["best_rightness"])
                    else event["best_rightness"]
                )

                delta_r = current_rightness - event["last_rightness"]

                # Heuristic:
                # mouse_dx * delta_rightness < 0  => front side within hemisphere
                # mouse_dx * delta_rightness > 0  => back side within hemisphere
                if abs(mouse_dx) >= MOVE_MIN_PIXELS and abs(delta_r) >= 0.015:
                    score = abs(mouse_dx * delta_r)

                    if mouse_dx * delta_r < 0:
                        event["front_score"] += score
                    else:
                        event["back_score"] += score

                event["last_rightness"] = current_rightness

                # If energy stays active, let event live slightly longer
                if current_rms > max(noise_rms * 1.6, ABS_MIN_RMS * 0.8):
                    event["until"] = max(event["until"], now + 0.05)

                payload = build_payload(event, current_rightness, current_ild_db, mouse_dx)
                publisher.send(payload)

                if now > event["until"]:
                    publisher.send({
                        "type": "direction",
                        "active": False,
                        "mode": "idle",
                        "side": event["side"],
                        "center_deg": 0,
                        "width_deg": 0,
                        "confidence": 0,
                        "ts": time.time(),
                    })
                    print("[SHOT] end")
                    event = None


if __name__ == "__main__":
    main()