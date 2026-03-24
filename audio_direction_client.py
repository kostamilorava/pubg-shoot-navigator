"""
PUBG Sound Direction Producer  –  Mouse-Correlation Method
==========================================================
Detects gunshots / loud sounds via spectral-flux onset detection,
then uses the player's mouse sweep to resolve front-vs-back:

    1.  Shot detected  →  we know LEFT or RIGHT from stereo ILD
    2.  Player sweeps mouse  →  stereo balance shifts
    3.  If turning RIGHT makes the sound move LEFT  →  it's in FRONT
        If turning RIGHT makes the sound move MORE RIGHT →  it's BEHIND
    4.  Accumulated correlation + ILD magnitude  →  360° angle estimate

Works with stereo / HRTF — no surround setup needed.

pip install numpy soundcard websocket-client pynput
"""

import json
import math
import time
import threading
from collections import deque

import numpy as np
import soundcard as sc
from websocket import create_connection
from pynput import mouse

# ------------------------------------------------------------------ #
#  CONFIG                                                              #
# ------------------------------------------------------------------ #
WS_URL = "ws://192.168.1.50:8080/ws?role=producer"
SAMPLE_RATE = 48000
BLOCK_SIZE = 2048          # ~43 ms per block — smoother ILD than 1024

# Onset detection
SHOT_MIN_GAP_MS   = 150
FLUX_TRIGGER_Z    = 3.2
RMS_TRIGGER_MULT  = 2.2
ABS_MIN_RMS       = 0.005

# Event timing
EVENT_HOLD_MS     = 1200   # keep event alive after last trigger
EVENT_MAX_MS      = 4000   # hard cap on event duration

# Rightness smoothing
RIGHTNESS_EMA     = 0.35   # higher = more responsive but noisier

# Mouse correlation
MOUSE_WINDOW_MS   = 120    # look-back window for mouse dx
MOVE_MIN_PX       = 8      # ignore tiny jitter
DELTA_R_MIN       = 0.012  # minimum rightness change to count as signal
CORR_MIN_SAMPLES  = 3      # need this many good samples before trusting front/back
CORR_MIN_SCORE    = 0.15   # minimum accumulated |score| before trusting

# Angle mapping
COARSE_WIDTH_DEG     = 150  # width when we only know left/right
REFINED_MIN_WIDTH    = 20
REFINED_MAX_WIDTH    = 80


# ------------------------------------------------------------------ #
#  MOUSE STATE                                                         #
# ------------------------------------------------------------------ #
mouse_events = deque(maxlen=6000)
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


def get_mouse_dx(window_ms=MOUSE_WINDOW_MS):
    """Sum of mouse dx over the last window_ms milliseconds."""
    cutoff = time.time() - window_ms / 1000.0
    total = 0.0
    with mouse_lock:
        while mouse_events and mouse_events[0][0] < cutoff - 2.0:
            mouse_events.popleft()
        for ts, dx in mouse_events:
            if ts >= cutoff:
                total += dx
    return total


# ------------------------------------------------------------------ #
#  WEBSOCKET PUBLISHER                                                 #
# ------------------------------------------------------------------ #
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

    def send(self, payload: dict):
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


# ------------------------------------------------------------------ #
#  AUDIO HELPERS                                                       #
# ------------------------------------------------------------------ #
def to_stereo(block):
    arr = np.asarray(block, dtype=np.float32)
    if arr.ndim == 1:
        return np.stack([arr, arr], axis=1)
    if arr.shape[1] > 2:
        return arr[:, :2]
    return arr


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def compute_rightness(stereo):
    """Returns (rightness in [-1,1], ild_db)."""
    l_rms = rms(stereo[:, 0])
    r_rms = rms(stereo[:, 1])
    ild_db = 20.0 * np.log10((r_rms + 1e-9) / (l_rms + 1e-9))
    rightness = float(np.tanh(ild_db / 8.0))
    return rightness, float(ild_db)


def spectral_flux(mono):
    windowed = mono * np.hanning(len(mono))
    return np.abs(np.fft.rfft(windowed))


def open_loopback():
    speaker = sc.default_speaker()
    print(f"[AUDIO] Default speaker: {speaker.name}")
    try:
        for mic in sc.all_microphones(include_loopback=True):
            name = getattr(mic, "name", str(mic))
            if speaker.name in name or name in speaker.name:
                print(f"[AUDIO] Using loopback mic: {name}")
                return mic.recorder(samplerate=SAMPLE_RATE, channels=2, blocksize=BLOCK_SIZE)
    except Exception as exc:
        print(f"[AUDIO] Loopback search failed: {exc}")
    print("[AUDIO] Falling back to speaker.recorder()")
    return speaker.recorder(samplerate=SAMPLE_RATE, channels=2, blocksize=BLOCK_SIZE)


# ------------------------------------------------------------------ #
#  DIRECTION FROM CORRELATION                                          #
# ------------------------------------------------------------------ #
def compute_angle(side, side_bias, is_front):
    """
    Map (side, magnitude, front/back) to 360° angle.
    0° = front, 90° = right, 180° = back, 270° = left.
    """
    lateral = min(1.0, side_bias)

    if side == "right":
        if is_front:
            angle = lateral * 90.0             # 0° → 90°
        else:
            angle = 180.0 - lateral * 90.0     # 180° → 90°
    else:
        if is_front:
            angle = 360.0 - lateral * 90.0     # 360° → 270°
        else:
            angle = 180.0 + lateral * 90.0     # 180° → 270°

    return angle % 360.0


def compute_width(has_front_back, confidence, separation):
    if not has_front_back:
        return COARSE_WIDTH_DEG
    narrowing = min(1.0, separation * 0.6 + confidence * 0.4)
    width = REFINED_MAX_WIDTH - narrowing * (REFINED_MAX_WIDTH - REFINED_MIN_WIDTH)
    return max(REFINED_MIN_WIDTH, int(width))


# ------------------------------------------------------------------ #
#  MAIN                                                                #
# ------------------------------------------------------------------ #
def main():
    start_mouse_listener()
    publisher = Publisher(WS_URL)

    prev_mag = None
    flux_mean = 1.0
    flux_dev = 1.0
    noise_rms = 0.003
    last_trigger_time = 0.0

    smooth_rightness = 0.0
    event = None

    print("[INFO] PUBG direction producer (mouse-correlation method)")
    print(f"[INFO] WS → {WS_URL}")
    print(f"[INFO] Block: {BLOCK_SIZE} samples ({BLOCK_SIZE / SAMPLE_RATE * 1000:.0f} ms)")
    print("[INFO] Workflow: shot detected → sweep mouse → front/back resolved")

    with open_loopback() as recorder:
        while True:
            block = recorder.record(numframes=BLOCK_SIZE)
            now = time.time()

            stereo = to_stereo(block)
            mono = stereo.mean(axis=1)

            current_rms = rms(mono)
            raw_rightness, ild_db = compute_rightness(stereo)

            # EMA-smooth rightness to reduce per-block noise
            smooth_rightness += RIGHTNESS_EMA * (raw_rightness - smooth_rightness)

            # Spectral flux onset detection
            mag = spectral_flux(mono)
            if prev_mag is None:
                prev_mag = mag
                continue

            flux = float(np.maximum(mag - prev_mag, 0.0).sum())
            prev_mag = mag

            flux_z = (flux - flux_mean) / (flux_dev + 1e-6)
            flux_mean = 0.98 * flux_mean + 0.02 * flux
            flux_dev = 0.98 * flux_dev + 0.02 * max(1.0, abs(flux - flux_mean))

            if event is None:
                noise_rms = 0.995 * noise_rms + 0.005 * current_rms

            # --- Onset trigger ---
            shot_trigger = (
                flux_z > FLUX_TRIGGER_Z
                and current_rms > max(noise_rms * RMS_TRIGGER_MULT, ABS_MIN_RMS)
                and (now - last_trigger_time) > SHOT_MIN_GAP_MS / 1000.0
            )

            side = "right" if smooth_rightness >= 0 else "left"
            mouse_dx = get_mouse_dx()

            if shot_trigger:
                last_trigger_time = now

                if event is None:
                    event = {
                        "started":        now,
                        "until":          now + EVENT_HOLD_MS / 1000.0,
                        "side":           side,
                        "prev_rightness": smooth_rightness,
                        "best_rightness": smooth_rightness,
                        "front_score":    0.0,
                        "back_score":     0.0,
                        "corr_samples":   0,
                    }
                    print(f"[SHOT] detected  side={side}  "
                          f"rightness={smooth_rightness:+.3f}  "
                          f"ild={ild_db:+.1f}dB  "
                          f"rms_ratio={current_rms / noise_rms:.1f}x")
                else:
                    event["until"] = max(event["until"], now + EVENT_HOLD_MS / 1000.0)

            # --- Process active event ---
            if event is not None:
                # Track strongest side bias seen during event
                if abs(smooth_rightness) > abs(event["best_rightness"]):
                    event["best_rightness"] = smooth_rightness

                # --- Mouse-audio correlation ---
                delta_r = smooth_rightness - event["prev_rightness"]
                event["prev_rightness"] = smooth_rightness

                if abs(mouse_dx) >= MOVE_MIN_PX and abs(delta_r) >= DELTA_R_MIN:
                    # mouse_dx * delta_r < 0  →  FRONT (turning right shifts sound left)
                    # mouse_dx * delta_r > 0  →  BACK  (turning right shifts sound right)
                    product = mouse_dx * delta_r
                    score = min(abs(product), 50.0)  # cap outliers

                    if product < 0:
                        event["front_score"] += score
                    else:
                        event["back_score"] += score

                    event["corr_samples"] += 1

                # Keep alive if audio still present
                if current_rms > max(noise_rms * 1.5, ABS_MIN_RMS * 0.8):
                    event["until"] = max(event["until"], now + 0.06)

                # Hard cap
                if now - event["started"] > EVENT_MAX_MS / 1000.0:
                    event["until"] = min(event["until"], now)

                # --- Build direction estimate ---
                side_bias = min(1.0, abs(event["best_rightness"]))
                total_corr = event["front_score"] + event["back_score"]
                has_front_back = (
                    event["corr_samples"] >= CORR_MIN_SAMPLES
                    and total_corr >= CORR_MIN_SCORE
                )

                if has_front_back:
                    is_front = event["front_score"] >= event["back_score"]
                    separation = abs(event["front_score"] - event["back_score"]) / (total_corr + 1e-9)
                    confidence = min(1.0, 0.5 + separation * 0.3 + min(0.2, total_corr * 0.02))
                    mode = "refined"
                else:
                    is_front = True
                    separation = 0.0
                    confidence = min(0.4, 0.15 + side_bias * 0.25)
                    mode = "coarse"

                center = compute_angle(event["side"], side_bias, is_front)
                width = compute_width(has_front_back, confidence, separation)

                publisher.send({
                    "type":         "direction",
                    "active":       True,
                    "mode":         mode,
                    "side":         event["side"],
                    "center_deg":   round(center, 1),
                    "width_deg":    width,
                    "confidence":   round(confidence, 3),
                    "rightness":    round(smooth_rightness, 4),
                    "ild_db":       round(ild_db, 2),
                    "mouse_dx":     round(mouse_dx, 1),
                    "front_score":  round(event["front_score"], 3),
                    "back_score":   round(event["back_score"], 3),
                    "corr_samples": event["corr_samples"],
                    "ts":           now,
                })

                # --- Expire ---
                if now > event["until"]:
                    f = event["front_score"]
                    b = event["back_score"]
                    fb = "FRONT" if f >= b else "BACK" if has_front_back else "?"
                    print(f"[SHOT] ended     angle={center:5.1f}°  {fb}  "
                          f"front={f:.2f} back={b:.2f}  "
                          f"samples={event['corr_samples']}  conf={confidence:.0%}")

                    publisher.send({
                        "type":       "direction",
                        "active":     False,
                        "mode":       "idle",
                        "side":       event["side"],
                        "center_deg": 0,
                        "width_deg":  0,
                        "confidence": 0,
                        "ts":         now,
                    })
                    event = None


if __name__ == "__main__":
    main()