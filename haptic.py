"""
Haptic Visualizer — Improved, cleaned version

Features:
 - Programmatic human silhouette (no PNG) with accurate, proportional zones
 - Zones anchored in relative coordinates [0..1] on the body
 - Draggable zone markers (save/load mapping to JSON)
 - Play/Pause, Smoothing, Threshold, Calibrate, Preset, Save/Load, Learn controls
 - Clicking a zone opens an analytics figure with 3 labeled subplots:
     1) Normalized band energy over time (x: seconds, y: normalized energy)
     2) Activation histogram (x: normalized energy, y: counts)
     3) Spectrogram (x: seconds, y: frequency Hz)
 - Uses STFT per-channel energies to map audio to zones
 - No CSV/MP4 export; no emojis in UI

Requirements:
  numpy scipy matplotlib soundfile
"""

import os
import json
import time
import math
import tempfile
from datetime import datetime

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from matplotlib.widgets import Button, Slider
import soundfile as sf

# Optional librosa (nicer spectrogram) — use if available
try:
    import librosa
    import librosa.display
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

# -------------------------- CONFIG ----------------------------------------
AUDIO_FILE = "test.mp3"
ZONES_JSON = "zones_user.json"

N_FFT = 2048
HOP_LENGTH = 1024

FIG_BG = "#0d0d0d"
BODY_FILL = "#fff6f6"
BODY_EDGE = "#333333"
CIRCLE_BASE_SIZE = 600
SMOOTH_ALPHA = 0.6
THRESH_DEFAULT = 0.02
COLORMAP = "plasma"

# default zones: positions are relative coordinates inside the body axes (x in [0,1], y in [0,1])
DEFAULT_ZONES = {
    "ankle_L": {"pos": [0.40, 0.10], "band": [20, 100], "label": "Ankle L", "gain": 1.0},
    "ankle_R": {"pos": [0.60, 0.10], "band": [20, 100], "label": "Ankle R", "gain": 1.0},
    "hip_L":   {"pos": [0.43, 0.28], "band": [60, 200], "label": "Hip L", "gain": 1.0},
    "hip_R":   {"pos": [0.57, 0.28], "band": [60, 200], "label": "Hip R", "gain": 1.0},
    "chest_L": {"pos": [0.46, 0.54], "band": [100, 300], "label": "Chest L", "gain": 1.0},
    "chest_R": {"pos": [0.54, 0.54], "band": [100, 300], "label": "Chest R", "gain": 1.0},
    "arm_L":   {"pos": [0.25, 0.60], "band": [300, 1000], "label": "Arm L", "gain": 1.0},
    "arm_R":   {"pos": [0.75, 0.60], "band": [300, 1000], "label": "Arm R", "gain": 1.0},
    "hand_L":  {"pos": [0.18, 0.78], "band": [1000, 4000], "label": "Hand L", "gain": 1.0},
    "hand_R":  {"pos": [0.82, 0.78], "band": [1000, 4000], "label": "Hand R", "gain": 1.0},
}

# ------------------------ Utilities ---------------------------------------
def ensure_file_exists(path, name="File"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")

def load_zones(json_path, fallback=DEFAULT_ZONES):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            d = json.load(f)
        # ensure gain present
        for k in d:
            if "gain" not in d[k]:
                d[k]["gain"] = 1.0
        return d
    else:
        with open(json_path, "w") as f:
            json.dump(fallback, f, indent=2)
        # return a deep copy
        return json.loads(json.dumps(fallback))

def save_zones(json_path, zones):
    with open(json_path, "w") as f:
        json.dump(zones, f, indent=2)
    print(f"Saved zones to {json_path}")

def load_audio(path):
    ensure_file_exists(path, "Audio file")
    y, sr = sf.read(path, always_2d=True)
    left = y[:,0].astype(np.float32)
    right = left.copy() if y.shape[1] == 1 else y[:,1].astype(np.float32)
    mono = 0.5*(left + right)
    return left, right, mono, sr

def stft_abs(x, sr):
    f, t, Z = signal.stft(x, fs=sr, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH, boundary=None)
    return f, t, np.abs(Z)

def band_energy_mean(S, freqs, low, high):
    idx = np.where((freqs >= low) & (freqs < high))[0]
    if idx.size == 0:
        return np.zeros(S.shape[1])
    return np.mean(S[idx,:], axis=0)

def smooth_arr(arr, alpha=SMOOTH_ALPHA):
    a = 1.0 - alpha
    if arr.size == 0:
        return arr
    sm = np.zeros_like(arr)
    sm[0] = arr[0]
    for i in range(1, len(arr)):
        sm[i] = (1-a) * sm[i-1] + a * arr[i]
    return sm

# ---------------------- Haptic Visualizer ---------------------------------
class HapticVisualizer:
    def __init__(self, audio_path=AUDIO_FILE, zones_json=ZONES_JSON):
        self.audio_path = audio_path
        self.zones_json = zones_json
        self.zones = load_zones(zones_json)
        self.left, self.right, self.mono, self.sr = load_audio(audio_path)

        # STFTs for both channels
        self.freqs, self.times, self.S_left = stft_abs(self.left, self.sr)
        _, _, self.S_right = stft_abs(self.right, self.sr)
        self.frame_count = self.S_left.shape[1]
        self.frame_rate = self.sr / HOP_LENGTH
        self.duration = len(self.mono) / float(self.sr)

        self.smoothing = SMOOTH_ALPHA
        self.threshold = THRESH_DEFAULT

        # compute energies
        self.energies = self._compute_all_zone_energies()
        self.normalized = self._normalize_and_smooth(self.energies, self.smoothing)

        # Matplotlib UI
        plt.rcParams['figure.facecolor'] = FIG_BG
        self.fig = plt.figure(figsize=(14, 8), facecolor=FIG_BG, constrained_layout=False)
        # Use GridSpec to make stable layout
        gs = self.fig.add_gridspec(1, 5, width_ratios=[3,0,0,0,2], wspace=0.05)
        self.ax_body = self.fig.add_subplot(gs[0,0])
        self.ax_body.set_facecolor("white")
        self.ax_body.set_xlim(0,1); self.ax_body.set_ylim(0,1)
        self.ax_body.set_xticks([]); self.ax_body.set_yticks([])
        self.ax_body.set_title("Haptic Body Map — drag markers to reposition", color="black", fontsize=14)

        # controls go in the wide right-most column (gs[:,4])
        self.ax_control = self.fig.add_subplot(gs[:,4])
        self.ax_control.axis("off")
        self._build_controls_layout()

        # draw body (programmatically)
        self._draw_body_shape()

        # init markers
        self.markers = {}
        self._init_markers()

        # colormap
        self.cmap = plt.get_cmap(COLORMAP)
        self.norm = plt.Normalize(vmin=0, vmax=1)

        # interactive dragging
        self._dragging = {"zone": None}
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

        # analytics click
        self.fig.canvas.mpl_connect("button_press_event", self._on_click_for_analytics)

        # animation
        from matplotlib.animation import FuncAnimation
        self.state = {"playing": False, "start_time": None, "frame": 0}
        self.anim = FuncAnimation(self.fig, self._update_frame, frames=self.frame_count, interval=1000/self.frame_rate, blit=False)

    # -------------------- UI: controls layout -----------------------------
    def _build_controls_layout(self):
        # We'll place controls vertically inside ax_control using relative coords
        self.control_elements = {}
        y = 0.94
        spacing = 0.07

        def add_button(label, callback):
            nonlocal y
            axb = self.fig.add_axes([0.72, y-0.03, 0.22, 0.05])  # positions chosen to live within right column area
            btn = Button(axb, label)
            btn.on_clicked(callback)
            y -= spacing
            return btn

        def add_slider(label, vmin, vmax, init, callback):
            nonlocal y
            axs = self.fig.add_axes([0.72, y-0.03, 0.22, 0.03])
            s = Slider(axs, label, vmin, vmax, valinit=init)
            s.on_changed(callback)
            y -= 0.06
            return s

        # Play/Pause
        self.btn_play = add_button("Play", self._play)
        self.btn_pause = add_button("Pause", self._pause)

        # Smoothing and threshold sliders
        self.s_smooth = add_slider("Smoothing", 0.0, 0.99, self.smoothing, self._set_smoothing)
        self.s_thresh = add_slider("Threshold", 0.0, 0.5, self.threshold, self._set_threshold)

        # Calibrate & Preset
        self.btn_cal = add_button("Calibrate Gains", self._calibrate_gains)
        self.btn_preset = add_button("Preset: Bass-Heavy", self._preset_bass_heavy)

        # Save / Load
        self.btn_save = add_button("Save Map", self._save_map)
        self.btn_load = add_button("Load Map", self._load_map)

        # Learn mode (educational)
        self.btn_learn = add_button("Learn about mapping", self._learn_mode)

        # Per-zone gain sliders (stack)
        # We'll create a vertical stack; if many zones exist, stop when hitting bottom to avoid overlap.
        y -= 0.02
        self.zone_gain_sliders = {}
        for zone in sorted(self.zones.keys()):
            if y < 0.06:
                break
            axs = self.fig.add_axes([0.72, y-0.03, 0.22, 0.03])
            slider = Slider(axs, zone, 0.2, 4.0, valinit=float(self.zones[zone].get("gain", 1.0)))
            slider.on_changed(self._make_gain_callback(zone))
            self.zone_gain_sliders[zone] = slider
            y -= 0.047

        # Info text area (top right, anchored)
        self.info_text_pos = (0.72, 0.04)
        self.info_text = self.fig.text(self.info_text_pos[0], self.info_text_pos[1], "", color="white", fontsize=9)

    # -------------------- Programmatic body -------------------------------
    def _draw_body_shape(self):
        ax = self.ax_body
        # clean, symmetrical silhouette using polygons and ellipses
        # torso
        torso_pts = [
            (0.48, 0.72), (0.52, 0.72), (0.62, 0.45),
            (0.57, 0.30), (0.50, 0.26), (0.43, 0.30), (0.38, 0.45)
        ]
        torso = Polygon(torso_pts, closed=True, edgecolor=BODY_EDGE, facecolor=BODY_FILL, linewidth=1.2, zorder=1)
        ax.add_patch(torso)
        # head
        head = Ellipse((0.50, 0.88), 0.14, 0.16, facecolor=BODY_FILL, edgecolor=BODY_EDGE, linewidth=1.0, zorder=1)
        ax.add_patch(head)
        # left arm
        left_arm = Polygon([(0.38,0.70),(0.22,0.58),(0.26,0.55),(0.40,0.66)], closed=True, facecolor=BODY_FILL, edgecolor=BODY_EDGE, zorder=1)
        right_arm = Polygon([(0.62,0.70),(0.78,0.58),(0.74,0.55),(0.60,0.66)], closed=True, facecolor=BODY_FILL, edgecolor=BODY_EDGE, zorder=1)
        ax.add_patch(left_arm); ax.add_patch(right_arm)
        # legs
        left_leg = Polygon([(0.47,0.26),(0.46,0.05),(0.50,0.05),(0.51,0.26)], closed=True, facecolor=BODY_FILL, edgecolor=BODY_EDGE, zorder=1)
        right_leg = Polygon([(0.51,0.26),(0.54,0.05),(0.58,0.05),(0.55,0.26)], closed=True, facecolor=BODY_FILL, edgecolor=BODY_EDGE, zorder=1)
        ax.add_patch(left_leg); ax.add_patch(right_leg)

    # -------------------- Markers ----------------------------------------
    def _init_markers(self):
        for zone_name, info in self.zones.items():
            x,y = info["pos"]
            color = (0.45, 0.35, 0.75, 0.85) if zone_name.endswith("_L") else (0.30,0.60,0.9,0.85)
            sc = self.ax_body.scatter([x], [y], s=CIRCLE_BASE_SIZE, facecolors=[color], edgecolors='k', linewidths=0.7, zorder=10)
            ann = self.ax_body.text(x, y+0.03, info.get("label", zone_name), ha='center', fontsize=9, zorder=11)
            self.markers[zone_name] = {"scatter": sc, "ann": ann}
        self.fig.canvas.draw_idle()

    # -------------------- Energies computation ---------------------------
    def _compute_all_zone_energies(self):
        energies = {}
        for z, info in self.zones.items():
            low, high = info["band"]
            if z.endswith("_L"):
                energies[z] = band_energy_mean(self.S_left, self.freqs, low, high)
            elif z.endswith("_R"):
                energies[z] = band_energy_mean(self.S_right, self.freqs, low, high)
            else:
                energies[z] = 0.5 * (band_energy_mean(self.S_left, self.freqs, low, high) + band_energy_mean(self.S_right, self.freqs, low, high))
        return energies

    def _normalize_and_smooth(self, energies, smoothing):
        gmax = max((np.max(v) if v.size else 0.0) for v in energies.values()) or 1.0
        out = {}
        for k, arr in energies.items():
            n = arr / gmax
            out[k] = smooth_arr(n, alpha=smoothing)
        return out

    # -------------------- Controls callbacks ----------------------------
    def _play(self, event=None):
        try:
            from IPython.display import Audio, display
            display(Audio(self.mono, rate=self.sr, autoplay=True))
            self.state["playing"] = True
            self.state["start_time"] = time.time() - (self.state["frame"] / self.frame_rate)
            self._info("Playing (in-notebook)")
        except Exception:
            tmp = os.path.join(tempfile.gettempdir(), f"__tmp_{int(time.time())}.wav")
            sf.write(tmp, self.mono, self.sr)
            self.state["playing"] = True
            self.state["start_time"] = time.time() - (self.state["frame"] / self.frame_rate)
            self._info(f"Playing external audio (written to tmp file): {tmp}")

    def _pause(self, event=None):
        self.state["playing"] = False
        self._info("Paused")

    def _set_smoothing(self, val):
        self.smoothing = float(val)
        self.normalized = self._normalize_and_smooth(self.energies, self.smoothing)
        self._info(f"Smoothing set to {self.smoothing:.2f}")

    def _set_threshold(self, val):
        self.threshold = float(val)
        self._info(f"Threshold set to {self.threshold:.3f}")

    def _make_gain_callback(self, zone):
        def cb(v):
            self.zones[zone]["gain"] = float(v)
            self._info(f"Gain {zone} = {v:.2f}")
        return cb

    def _save_map(self, event=None):
        save_zones(self.zones_json, self.zones)
        self._info("Map saved")

    def _load_map(self, event=None):
        self.zones = load_zones(self.zones_json)
        # update markers and sliders
        for z, info in self.zones.items():
            if z in self.markers:
                x,y = info["pos"]
                sc = self.markers[z]["scatter"]
                sc.set_offsets([[x,y]])
                self.markers[z]["ann"].set_position((x, y+0.03))
            if z in self.zone_gain_sliders:
                self.zone_gain_sliders[z].set_val(info.get("gain", 1.0))
        self.energies = self._compute_all_zone_energies()
        self.normalized = self._normalize_and_smooth(self.energies, self.smoothing)
        self._info("Map loaded and energies recomputed")

    def _preset_bass_heavy(self, event=None):
        for z in self.zones:
            if "ankle" in z or "hip" in z:
                self.zones[z]["gain"] = 1.8
            else:
                self.zones[z]["gain"] = 1.0
            if z in self.zone_gain_sliders:
                self.zone_gain_sliders[z].set_val(self.zones[z]["gain"])
        self._info("Applied preset: Bass-Heavy")

    def _learn_mode(self, event=None):
        # Show an educational text figure describing mapping ideas
        txt = (
            "Learn about Haptic Music Mapping\n\n"
            " - Low frequencies (e.g., bass, kick) are strong felt sensations. Map to hips/ankles.\n"
            " - Mid frequencies (e.g., body of instruments) can be mapped to chest/torso.\n"
            " - High frequencies (e.g., melody/timbre) are best on arms/hands for local detail.\n\n"
            "Calibration: use 'Calibrate Gains' to equalize peak energies across zones.\n"
            "Click a zone to view analytics: Energy over time, Distribution, and Spectrogram."
        )
        fig = plt.figure(figsize=(6,4))
        fig.patch.set_facecolor("#111111")
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.01, 0.99, txt, va="top", ha="left", color="white", fontsize=10, wrap=True)
        plt.show()

    # -------------------- Calibration -----------------------------------
    def _calibrate_gains(self, event=None):
        # Compute peaks per zone (use normalized energies) and recommend gains to match median peak
        peaks = {}
        for z, arr in self.normalized.items():
            peaks[z] = float(np.max(arr)) if arr.size else 0.0
        target = float(np.median(list(peaks.values()))) or 1.0
        recommended = {}
        for z,p in peaks.items():
            recommended[z] = float(1.0 if p <= 0 else max(0.25, min(4.0, target / p)))
            self.zones[z]["gain"] = recommended[z]
            if z in self.zone_gain_sliders:
                self.zone_gain_sliders[z].set_val(recommended[z])
        lines = ["Calibration recommendations (gain):"] + [f" {z}: peak={peaks[z]:.3f} -> gain={recommended[z]:.2f}" for z in sorted(recommended)]
        self._info("Calibration applied: per-zone gains updated")
        print("\n".join(lines))

    # -------------------- Mouse interactions (drag) ----------------------
    def _on_press(self, event):
        if event.inaxes != self.ax_body:
            return
        x,y = event.xdata, event.ydata
        best, bestd = None, 1e9
        for z, item in self.markers.items():
            ox, oy = item["scatter"].get_offsets()[0]
            d = math.hypot(x-ox, y-oy)
            if d < bestd:
                bestd, best = d, z
        # threshold for selecting marker (relative)
        if best is not None and bestd < 0.12:
            self._dragging["zone"] = best
            self._info(f"Drag start: {best}")

    def _on_move(self, event):
        if self._dragging["zone"] is None:
            return
        if event.inaxes != self.ax_body:
            return
        z = self._dragging["zone"]
        x,y = event.xdata, event.ydata
        # clamp coordinates in [0,1]
        x = min(max(0.02, x), 0.98)
        y = min(max(0.02, y), 0.98)
        self.zones[z]["pos"] = [x,y]
        sc = self.markers[z]["scatter"]
        sc.set_offsets([[x,y]])
        self.markers[z]["ann"].set_position((x, y+0.03))
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if self._dragging["zone"] is not None:
            z = self._dragging["zone"]
            self._dragging["zone"] = None
            save_zones(self.zones_json, self.zones)
            self._info(f"Placed {z} at {tuple(self.zones[z]['pos'])}")

    # -------------------- Click analytics --------------------------------
    def _on_click_for_analytics(self, event):
        # If click in body and within a marker radius, open analytics
        if event.inaxes != self.ax_body:
            return
        x,y = event.xdata, event.ydata
        best, bestd = None, 1e9
        for z, item in self.markers.items():
            ox, oy = item["scatter"].get_offsets()[0]
            d = math.hypot(x-ox, y-oy)
            if d < bestd:
                bestd, best = d, z
        if best is not None and bestd < 0.12:
            # Distinguish between drag start vs click: if not dragging, open analytics
            if self._dragging["zone"] is None:
                self._open_zone_analytics(best)

    def _open_zone_analytics(self, zone):
        # produce labeled analytics figure for the selected zone
        arr = self.normalized[zone] if zone in self.normalized else np.zeros(len(self.times))
        t = self.times
        fig, axes = plt.subplots(3,1, figsize=(8,7))
        fig.suptitle(f"Zone Analytics — {zone}", fontsize=12)

        # 1. Energy over time
        axes[0].plot(t, arr, color='tab:purple')
        axes[0].set_title("1) Normalized Band Energy Over Time")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Normalized energy (0-1)")
        axes[0].grid(True)

        # 2. Activation histogram
        axes[1].hist(arr, bins=50, color='tab:orange', edgecolor='black', linewidth=0.2)
        mean, std, peak = float(np.mean(arr)), float(np.std(arr)), float(np.max(arr))
        axes[1].set_title("2) Activation Distribution (histogram)")
        axes[1].set_xlabel("Normalized energy")
        axes[1].set_ylabel("Count")
        axes[1].text(0.02, 0.85, f"Mean: {mean:.4f}\nStd: {std:.4f}\nPeak: {peak:.4f}", transform=axes[1].transAxes, bbox=dict(facecolor='white', alpha=0.6))

        # 3. Spectrogram
        axes[2].set_title("3) Spectrogram (Left channel)")
        if HAS_LIBROSA:
            D = np.abs(librosa.stft(self.left))
            librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=self.sr, ax=axes[2], cmap='magma', y_axis='log', x_axis='time')
            axes[2].set_ylabel("Frequency (Hz)")
        else:
            axes[2].specgram(self.left, NFFT=1024, Fs=self.sr, noverlap=512, cmap='magma')
            axes[2].set_xlabel("Time (s)")
            axes[2].set_ylabel("Frequency (Hz)")
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.show()

    # -------------------- Update animation --------------------------------
    def _update_frame(self, idx):
        # sync with audio when playing
        if self.state["playing"] and self.state["start_time"] is not None:
            t_elapsed = time.time() - self.state["start_time"]
            idx = int(t_elapsed * self.frame_rate)
        idx = min(max(0, idx), self.frame_count - 1)
        self.state["frame"] = idx

        vals = {}
        for z in self.normalized:
            v = float(self.normalized[z][idx]) if self.normalized[z].size else 0.0
            v *= float(self.zones[z].get("gain", 1.0))
            vals[z] = v if v >= self.threshold else 0.0

        # update markers appearance according to activation
        for z, item in self.markers.items():
            sc = item["scatter"]
            v = min(1.0, vals.get(z, 0.0))
            rgba = self.cmap(self.norm(v))
            sc.set_facecolor(rgba)
            sc.set_alpha(0.45 + 0.5 * v)
            sc.set_sizes([CIRCLE_BASE_SIZE * (0.6 + 0.8 * v)])
        # info summary
        top = sorted(vals.items(), key=lambda x: -x[1])[:3]
        sample_idx = int(idx * HOP_LENGTH)
        halfwin = int(self.sr * 0.5)
        w0 = max(0, sample_idx - halfwin//2)
        w1 = min(len(self.mono), sample_idx + halfwin//2)
        rms = float(np.sqrt(np.mean(self.mono[w0:w1]**2))) if w1 > w0 else 0.0
        lines = [
            f"Time: {self.times[idx]:.2f}s / {self.duration:.2f}s",
            f"Frame: {idx}/{self.frame_count}",
            f"RMS (0.5s): {rms:.4f}",
            "Top zones:"
        ]
        for z,v in top:
            lines.append(f"  {z}: {v:.3f} (gain={self.zones[z].get('gain',1.0):.2f})")
        ts = datetime.now().strftime("%H:%M:%S")
        self.info_text.set_text(f"[{ts}] " + " | ".join(lines[:3]))
        # draw update
        self.fig.canvas.draw_idle()
        return []

    # -------------------- Utility: set info -------------------------------
    def _info(self, txt):
        ts = datetime.now().strftime("%H:%M:%S")
        self.info_text.set_text(f"[{ts}] {txt}")
        self.fig.canvas.draw_idle()

    # -------------------- Show -------------------------------------------
    def show(self):
        plt.show()

# -------------------------- MAIN ----------------------------------------
def main(audio=AUDIO_FILE, zones_json=ZONES_JSON):
    if not os.path.exists(audio):
        print(f"Audio file not found: {audio}")
        return
    vis = HapticVisualizer(audio, zones_json)
    print("Haptic Visualizer initialized.")
    print(" - Drag markers on the body to reposition zones.")
    print(" - Click a marker (without dragging) to open analytics (3 labeled plots).")
    print(" - Use 'Calibrate Gains' to equalize peak energies across zones.")
    vis.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Haptic Visualizer (improved)")
    parser.add_argument("--audio", type=str, default=AUDIO_FILE, help="Path to audio file (wav/mp3)")
    parser.add_argument("--zones", type=str, default=ZONES_JSON, help="Zones JSON file")
    args = parser.parse_args()
    main(audio=args.audio, zones_json=args.zones)
