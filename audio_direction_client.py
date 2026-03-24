import json
import time
import threading
from collections import deque

import numpy as np
import soundcard as sc
from pynput import mouse
from websocket import create_connection


# ============================================================
# CONFIG
# ============================================================

WS_URL = "ws://192.168.1.50:8080/ws?role=producer"   # <-- change this
SAMPLE_RATE = 48000
BLOCK_SIZE = 1024

# If left/right is reversed, set this to True
SWAP_CHANNELS = False

# Detection / timing
SHOT_MIN_GAP_MS = 180
EVENT_HOLD_MS = 900
MOTION_WINDOW_MS = 70
MOVE_MIN_PIXELS = 6

# Trigger sensitivity
FLUX_TRIGGER_Z = 3.5
RMS_TRIGGER_MULT = 2.4
ABS_MIN_RMS = 0.006

# Sector sizes
COARSE_WIDTH_DEG = 160
MIN_REFINED_WIDTH_DEG = 24

# Idle publish rate, so UI does not look dead
IDLE_HEARTBEAT_MS = 500


# ============================================================
# MOUSE STATE
# ============================================================

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
        while mouse_events and mouse_events[0][0] < cutoff - 1.0:
            mouse_events.popleft()

        for ts, dx in mouse_events:
            if ts >= cutoff:
                total += dx

    return total


# ============================================================
# WEBSOCKET PUBLISHER
# ============================================================

class Publisher:
    def __init__(self, url: str):
        self.url = url
        self.ws = None
        self.last_try = 0.0

    def ensure(self) -> bool:
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

    def send(self, payload: dict) -> None:
        if not self.ensure():
            return

        try:
            self.ws.send(json.dumps(payload))
        except Exception as exc:
            print(f"[WS] Send failed: {exc}")
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None


# ============================================================
# AUDIO HELPERS
# ============================================================

def to_stereo(block) -> np.ndarray:
    arr = np.asarray(block, dtype=np.float32)

    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=1)

    if arr.shape[1] > 2:
        arr = arr[:, :2]

    if SWAP_CHANNELS:
        arr = arr[:, ::-1]

    return arr


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def compute_rightness(stereo: np.ndarray):
    """
    Returns:
        rightness: [-1, 1]
        ild_db: interaural level difference in dB (R vs L)
    """
    left = stereo[:, 0]
    right = stereo[:, 1]

    l_rms = rms(left)
    r_rms = rms(right)

    ild_db = 20.0 * np.log10((r_rms + 1e-9) / (l_rms + 1e-9))

    # Smoothly squash dB ratio into [-1, 1]
    rightness = float(np.tanh(ild_db / 8.0))
    return rightness, float(ild_db)


def spectral_flux_mag(mono: np.ndarray) -> np.ndarray:
    windowed = mono * np.hanning(len(mono))
    mag = np.abs(np.fft.rfft(windowed))
    return mag


def open_loopback_recorder():
    """
    Best effort: try to capture system output / loopback.
    Works best on Windows.
    """
    try:
        default_speaker = sc.default_speaker()
        print(f"[AUDIO] Default speaker: {default_speaker.name}")
    except Exception as exc:
        raise RuntimeError(f"Could not get default speaker: {exc}") from exc

    try:
        loopbacks = sc.all_microphones(include_loopback=True)
        for mic in loopbacks:
            mic_name = getattr(mic, "name", str(mic))
            if default_speaker.name in mic_name or mic_name in default_speaker.name:
                print(f"[AUDIO] Using loopback device: {mic_name}")
                return mic.recorder(
                    samplerate=SAMPLE_RATE,
                    channels=2,
                    blocksize=BLOCK_SIZE,
                )
    except Exception as exc:
        print(f"[AUDIO] Loopback scan failed: {exc}")

    # fallback
    try:
        default_mic = sc.default_microphone()
        print(f"[AUDIO] Loopback not found, falling back to microphone: {default_mic.name}")
        return default_mic.recorder(
            samplerate=SAMPLE_RATE,
            channels=2,
            blocksize=BLOCK_SIZE,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not open loopback or microphone recorder. "
            "You likely need a working loopback device."
        ) from exc


# ============================================================
# DIRECTION / PAYLOAD
# ============================================================

def normalize_deg(deg: float) -> float:
    return deg % 360.0


def build_payload(event: dict, current_rightness: float, current_ild_db: float, mouse_dx: float) -> dict:
    side = event["side"]
    evidence = event["front_score"] + event["back_score"]
    frontish = event["front_score"] >= event["back_score"]

    side_bias = min(1.0, abs(event["best_rightness"]))

    if evidence < 0.8:
        # Only hemisphere is known
        center = 90.0 if side == "right" else 270.0
        width = max(100.0, COARSE_WIDTH_DEG - side_bias * 35.0)
        confidence = min(0.55, 0.22 + side_bias * 0.33)
        mode = "coarse"
    else:
        # Refine within the hemisphere
        separation = abs(event["front_score"] - event["back_score"]) / (evidence + 1e-9)
        confidence = min(1.0, 0.50 + separation * 0.50)
        mode = "refined"

        if side == "right" and frontish:
            # front-right: 0..90
            center = 15.0 + side_bias * 75.0
        elif side == "right" and not frontish:
            # back-right: 90..180
            center = 165.0 - side_bias * 75.0
        elif side == "left" and frontish:
            # front-left: 270..360
            center = normalize_deg(345.0 - side_bias * 75.0)
        else:
            # back-left: 180..270
            center = 195.0 + side_bias * 75.0

        width = max(
            MIN_REFINED_WIDTH_DEG,
            100.0 - confidence * 40.0 - min(28.0, evidence * 4.0),
        )

    start_deg = normalize_deg(center - width / 2.0)
    end_deg = normalize_deg(center + width / 2.0)

    return {
        "type": "direction",
        "active": True,
        "mode": mode,
        "side": side,
        "center_deg": float(normalize_deg(center)),
        "width_deg": float(width),
        "start_deg": float(start_deg),
        "end_deg": float(end_deg),
        "confidence": float(confidence),
        "rightness": float(current_rightness),
        "ild_db": float(current_ild_db),
        "mouse_dx": float(mouse_dx),
        "front_score": float(event["front_score"]),
        "back_score": float(event["back_score"]),
        "ts": time.time(),
    }


def idle_payload() -> dict:
    return {
        "type": "direction",
        "active": False,
        "mode": "idle",
        "side": "unknown",
        "center_deg": 0.0,
        "width_deg": 0.0,
        "start_deg": 0.0,
        "end_deg": 0.0,
        "confidence": 0.0,
        "rightness": 0.0,
        "ild_db": 0.0,
        "mouse_dx": 0.0,
        "front_score": 0.0,
        "back_score": 0.0,
        "ts": time.time(),
    }


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    start_mouse_listener()
    publisher = Publisher(WS_URL)

    prev_mag = None
    flux_mean = 1.0
    flux_dev = 1.0
    noise_rms = 0.002
    last_trigger_time = 0.0
    last_idle_publish = 0.0

    event = None

    print("[INFO] Starting audio direction producer...")
    print(f"[INFO] WS URL: {WS_URL}")
    print(f"[INFO] SWAP_CHANNELS={SWAP_CHANNELS}")

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

            flux_z = (flux - flux_mean) / (flux_dev + 1e-6)

            # Update rolling baseline
            flux_mean = 0.98 * flux_mean + 0.02 * flux
            flux_dev = 0.98 * flux_dev + 0.02 * max(1.0, abs(flux - flux_mean))

            if event is None:
                noise_rms = 0.995 * noise_rms + 0.005 * current_rms

            shot_trigger = (
                flux_z > FLUX_TRIGGER_Z
                and current_rms > max(noise_rms * RMS_TRIGGER_MULT, ABS_MIN_RMS)
                and (now - last_trigger_time) > (SHOT_MIN_GAP_MS / 1000.0)
            )

            mouse_dx = get_recent_mouse_dx()
            side = "right" if current_rightness >= 0 else "left"

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
                    print(
                        f"[SHOT] start | side={side} | "
                        f"rms={current_rms:.5f} | flux_z={flux_z:.2f} | "
                        f"rightness={current_rightness:.3f} | ild_db={current_ild_db:.2f}"
                    )
                else:
                    event["until"] = max(event["until"], now + (EVENT_HOLD_MS / 1000.0))

            if event is not None:
                # Keep strongest hemisphere evidence seen during event
                if abs(current_rightness) > abs(event["best_rightness"]):
                    event["best_rightness"] = current_rightness

                delta_r = current_rightness - event["last_rightness"]

                # Heuristic:
                # mouse_dx * delta_rightness < 0 => more frontish
                # mouse_dx * delta_rightness > 0 => more backish
                if abs(mouse_dx) >= MOVE_MIN_PIXELS and abs(delta_r) >= 0.015:
                    score = abs(mouse_dx * delta_r)

                    if mouse_dx * delta_r < 0:
                        event["front_score"] += score
                    else:
                        event["back_score"] += score

                event["last_rightness"] = current_rightness

                # Extend while sound still remains somewhat active
                if current_rms > max(noise_rms * 1.6, ABS_MIN_RMS * 0.8):
                    event["until"] = max(event["until"], now + 0.05)

                payload = build_payload(event, current_rightness, current_ild_db, mouse_dx)
                publisher.send(payload)

                if now > event["until"]:
                    print(
                        f"[SHOT] end | side={event['side']} | "
                        f"front_score={event['front_score']:.3f} | "
                        f"back_score={event['back_score']:.3f}"
                    )
                    publisher.send(idle_payload())
                    event = None

            else:
                # Heartbeat while idle, useful for UI/debug
                if now - last_idle_publish >= (IDLE_HEARTBEAT_MS / 1000.0):
                    publisher.send(idle_payload())
                    last_idle_publish = now


if __name__ == "__main__":
    main()