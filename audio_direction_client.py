"""
PUBG Sound Direction Producer  –  Stable Mouse-Correlation
==========================================================
pip install numpy soundcard websocket-client pynput scipy
"""

import json
import math
import time
import threading
from collections import deque

import numpy as np
from scipy.signal import butter, sosfilt
import soundcard as sc
from websocket import create_connection
from pynput import mouse

# ------------------------------------------------------------------ #
#  CONFIG                                                              #
# ------------------------------------------------------------------ #
WS_URL = "ws://192.168.1.50:8080/ws?role=producer"
SAMPLE_RATE = 48000
BLOCK_SIZE  = 2048          # ~43 ms per block

# Onset detection — LOWER thresholds to catch footsteps
SHOT_MIN_GAP_MS   = 100
FLUX_TRIGGER_Z    = 2.2     # lowered from 3.2 — catches quieter sounds
RMS_TRIGGER_MULT  = 1.8     # lowered from 2.2
ABS_MIN_RMS       = 0.003   # lowered from 0.005

# Event timing
EVENT_HOLD_MS     = 1400    # keep alive after last trigger
EVENT_MAX_MS      = 15000   # hard cap — auto fire can last many seconds

# ILD / rightness
RIGHTNESS_EMA     = 0.18    # slower smoothing = more stable (was 0.35)
OUTPUT_ANGLE_EMA  = 0.25    # smooth the output angle between frames
ILD_MIN_DB        = 1.2     # minimum |ILD| in dB to count as directional
                            # bullet impacts on your body are ~0 dB (centered)
                            # gunfire from a direction is typically 3–15 dB

# Mouse correlation
MOUSE_WINDOW_MS   = 150
MOVE_MIN_PX       = 10
DELTA_R_MIN       = 0.010
CORR_MIN_SAMPLES  = 4
CORR_MIN_SCORE    = 0.25

# Angle mapping
COARSE_WIDTH_DEG  = 150
REFINED_MIN_WIDTH = 20
REFINED_MAX_WIDTH = 80

# Bandpass for ILD — only frequencies where stereo separation is meaningful
#   Below ~1.5 kHz: ILD is near zero (wavelength > head size)
#   Above ~12 kHz:  less game audio content
ILD_LOW_HZ  = 1500
ILD_HIGH_HZ = 12000


# ------------------------------------------------------------------ #
#  BANDPASS FILTER                                                     #
# ------------------------------------------------------------------ #
def make_bandpass(low_hz, high_hz, sr, order=4):
    nyq = sr / 2.0
    low = max(low_hz / nyq, 0.001)
    high = min(high_hz / nyq, 0.999)
    return butter(order, [low, high], btype="band", output="sos")


BP_SOS = make_bandpass(ILD_LOW_HZ, ILD_HIGH_HZ, SAMPLE_RATE)


# ------------------------------------------------------------------ #
#  MOUSE STATE                                                         #
# ------------------------------------------------------------------ #
mouse_events = deque(maxlen=6000)
mouse_lock   = threading.Lock()
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
        self.ws  = None
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


def compute_rightness_bandpass(stereo):
    """
    Compute ILD only on bandpassed audio (1.5–12 kHz).
    This is where stereo separation actually exists in game audio.
    Low frequencies bleed equally into both channels → pure noise for ILD.
    """
    left_bp  = sosfilt(BP_SOS, stereo[:, 0])
    right_bp = sosfilt(BP_SOS, stereo[:, 1])

    l_rms = rms(left_bp)
    r_rms = rms(right_bp)

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
#  DIRECTION LOGIC                                                     #
# ------------------------------------------------------------------ #
def compute_angle(side, side_bias, is_front):
    """0° = front, 90° = right, 180° = back, 270° = left."""
    lateral = min(1.0, side_bias)
    if side == "right":
        angle = lateral * 90.0 if is_front else 180.0 - lateral * 90.0
    else:
        angle = (360.0 - lateral * 90.0) if is_front else (180.0 + lateral * 90.0)
    return angle % 360.0


def compute_width(has_front_back, confidence, separation):
    if not has_front_back:
        return COARSE_WIDTH_DEG
    narrowing = min(1.0, separation * 0.6 + confidence * 0.4)
    width = REFINED_MAX_WIDTH - narrowing * (REFINED_MAX_WIDTH - REFINED_MIN_WIDTH)
    return max(REFINED_MIN_WIDTH, int(width))


def smooth_angle(old_deg, new_deg, alpha):
    """
    EMA-smooth an angle, handling the 0°/360° wraparound correctly.
    """
    diff = (new_deg - old_deg + 180) % 360 - 180  # shortest arc
    return (old_deg + alpha * diff) % 360


# ------------------------------------------------------------------ #
#  MAIN                                                                #
# ------------------------------------------------------------------ #
def main():
    start_mouse_listener()
    publisher = Publisher(WS_URL)

    prev_mag        = None
    flux_mean       = 1.0
    flux_dev        = 1.0
    noise_rms       = 0.003
    last_trigger_time = 0.0

    smooth_rightness = 0.0
    output_angle     = 0.0      # smoothed output angle

    event = None

    print("[INFO] PUBG direction producer (stable mouse-correlation)")
    print(f"[INFO] WS → {WS_URL}")
    print(f"[INFO] Block: {BLOCK_SIZE} samples ({BLOCK_SIZE / SAMPLE_RATE * 1000:.0f} ms)")
    print(f"[INFO] ILD bandpass: {ILD_LOW_HZ}–{ILD_HIGH_HZ} Hz")

    with open_loopback() as recorder:
        while True:
            block = recorder.record(numframes=BLOCK_SIZE)
            now   = time.time()

            stereo = to_stereo(block)
            mono   = stereo.mean(axis=1)

            current_rms = rms(mono)
            raw_rightness, ild_db = compute_rightness_bandpass(stereo)

            # Slow EMA on rightness — stability over responsiveness
            smooth_rightness += RIGHTNESS_EMA * (raw_rightness - smooth_rightness)

            # Spectral flux onset
            mag = spectral_flux(mono)
            if prev_mag is None:
                prev_mag = mag
                continue

            flux = float(np.maximum(mag - prev_mag, 0.0).sum())
            prev_mag = mag

            flux_z = (flux - flux_mean) / (flux_dev + 1e-6)

            # Slow down flux adaptation during active events —
            # otherwise sustained auto fire becomes "normal" in ~2s
            # and the detector stops triggering
            if event is None:
                flux_alpha = 0.02        # normal rate
            else:
                flux_alpha = 0.002       # 10x slower during event
            flux_mean = (1 - flux_alpha) * flux_mean + flux_alpha * flux
            flux_dev  = (1 - flux_alpha) * flux_dev  + flux_alpha * max(1.0, abs(flux - flux_mean))

            if event is None:
                noise_rms = 0.995 * noise_rms + 0.005 * current_rms

            # --- Onset trigger ---
            shot_trigger = (
                flux_z > FLUX_TRIGGER_Z
                and current_rms > max(noise_rms * RMS_TRIGGER_MULT, ABS_MIN_RMS)
                and (now - last_trigger_time) > SHOT_MIN_GAP_MS / 1000.0
            )

            mouse_dx = get_mouse_dx()

            if shot_trigger:
                last_trigger_time = now

                if event is None:
                    side = "right" if smooth_rightness >= 0 else "left"
                    event = {
                        "started":          now,
                        "until":            now + EVENT_HOLD_MS / 1000.0,
                        "side":             side,            # LOCKED at start
                        "prev_rightness":   smooth_rightness,
                        "rightness_accum":  [smooth_rightness],  # accumulate samples
                        "front_score":      0.0,
                        "back_score":       0.0,
                        "corr_samples":     0,
                        "frozen_noise":     noise_rms,       # snapshot — doesn't drift
                    }
                    print(f"[SHOT] detected  side={side}  "
                          f"rightness={smooth_rightness:+.3f}  "
                          f"ild={ild_db:+.1f}dB  "
                          f"ratio={current_rms / noise_rms:.1f}x")
                else:
                    event["until"] = max(event["until"], now + EVENT_HOLD_MS / 1000.0)

            # --- Active event processing ---
            if event is not None:
                # How "directional" is this frame?
                abs_ild = abs(ild_db)
                is_directional = abs_ild >= ILD_MIN_DB

                # Only accumulate rightness from DIRECTIONAL frames.
                # Bullet impacts on your body are centered (~0 dB ILD)
                # and would wash out the actual gunfire direction.
                if is_directional:
                    weight = abs_ild  # stronger separation = more weight
                    event["rightness_accum"].append((smooth_rightness, weight))

                # --- Mouse-audio correlation ---
                # Also weight by directional strength — centered frames
                # contribute nothing to front/back resolution
                delta_r = smooth_rightness - event["prev_rightness"]
                event["prev_rightness"] = smooth_rightness

                if (abs(mouse_dx) >= MOVE_MIN_PX
                        and abs(delta_r) >= DELTA_R_MIN
                        and is_directional):
                    product = mouse_dx * delta_r
                    # Scale score by ILD strength — directional frames
                    # contribute more to the front/back decision
                    ild_weight = min(abs_ild / 3.0, 3.0)  # 3 dB → 1x, 9 dB → 3x
                    score = min(abs(product) * ild_weight, 80.0)

                    if product < 0:
                        event["front_score"] += score
                    else:
                        event["back_score"] += score
                    event["corr_samples"] += 1

                # --- Energy sustain ---
                fn = event["frozen_noise"]
                if current_rms > max(fn * 2.0, ABS_MIN_RMS):
                    event["until"] = max(event["until"], now + 0.30)
                elif current_rms > max(fn * 1.3, ABS_MIN_RMS * 0.5):
                    event["until"] = max(event["until"], now + 0.08)

                # Hard cap
                if now - event["started"] > EVENT_MAX_MS / 1000.0:
                    event["until"] = min(event["until"], now)

                # --- Compute stable direction ---
                # Weighted average of rightness — directional frames dominate,
                # centered impacts are excluded entirely
                acc = event["rightness_accum"]
                if len(acc) > 0:
                    values  = np.array([r for r, w in acc])
                    weights = np.array([w for r, w in acc])
                    w_total = weights.sum() + 1e-9
                    weighted_rightness = float((values * weights).sum() / w_total)
                else:
                    weighted_rightness = smooth_rightness

                side_bias = min(1.0, abs(weighted_rightness))

                # Re-evaluate side from accumulated evidence
                stable_side = "right" if weighted_rightness >= 0 else "left"
                if stable_side != event["side"]:
                    if abs(weighted_rightness) > 0.12 and len(acc) >= 6:
                        event["side"] = stable_side
                        print(f"[SHOT] side corrected → {stable_side} "
                              f"(weighted_r={weighted_rightness:+.3f})")

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

                raw_angle = compute_angle(event["side"], side_bias, is_front)
                width = compute_width(has_front_back, confidence, separation)

                # Smooth output angle
                output_angle = smooth_angle(output_angle, raw_angle, OUTPUT_ANGLE_EMA)

                publisher.send({
                    "type":         "direction",
                    "active":       True,
                    "mode":         mode,
                    "side":         event["side"],
                    "center_deg":   round(output_angle, 1),
                    "width_deg":    width,
                    "confidence":   round(confidence, 3),
                    "rightness":    round(smooth_rightness, 4),
                    "ild_db":       round(ild_db, 2),
                    "ild_abs":      round(abs_ild, 2),
                    "mouse_dx":     round(mouse_dx, 1),
                    "front_score":  round(event["front_score"], 3),
                    "back_score":   round(event["back_score"], 3),
                    "corr_samples": event["corr_samples"],
                    "dir_frames":   len(acc),
                    "ts":           now,
                })

                # --- Expire ---
                if now > event["until"]:
                    f = event["front_score"]
                    b = event["back_score"]
                    fb = "FRONT" if f >= b else "BACK" if has_front_back else "?"
                    print(f"[SHOT] ended     angle={output_angle:5.1f}°  {fb}  "
                          f"front={f:.2f} back={b:.2f}  "
                          f"samples={event['corr_samples']}  "
                          f"dir_frames={len(acc)}  "
                          f"weighted_r={weighted_rightness:+.3f}  "
                          f"conf={confidence:.0%}")

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