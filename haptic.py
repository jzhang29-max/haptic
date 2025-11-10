"""
Haptic Visualizer — autoplay, clickable analytics, working presets, cleaned UI.

Key features:
 - Auto-plays MP3 (uses soundfile or pydub fallback) via simpleaudio (non-blocking).
 - Programmatic human silhouette; markers anchored by relative coords.
 - Markers pulse with music amplitude/intensity. Clicking a marker opens 3 labeled plots:
    1) Left vs Right band energy over time (x = seconds, y = normalized energy)
    2) Activation histogram (x = normalized energy, y = count)
    3) Spectrogram (x = seconds, y = frequency Hz)
 - Working presets: Normal, Bass-Heavy, Treble-Heavy with explanation text shown.
 - Draggable markers (saved to JSON).
 - Minimal status updates (no continuously moving numbers).
"""

import os
import sys
import json
import math
import time
import tempfile
import threading
from datetime import datetime

import numpy as np
import pygame
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from matplotlib.widgets import Button, Slider
import soundfile as sf

# Optional: pydub to read mp3 fallback
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except Exception:
    HAS_PYDUB = False

# Playback library (preferred)
try:
    import simpleaudio as sa
    HAS_SIMPLEAUDIO = True
except Exception:
    HAS_SIMPLEAUDIO = False

# ---------------- Config / Your new zones ----------------
AUDIO_FILE = "test.mp3"
ZONES_JSON = "zones_user.json"

N_FFT = 2048
HOP_LENGTH = 1024

FIG_BG = "#0d0d0d"
BODY_FILL = "#fff6f6"
BODY_EDGE = "#333333"
CIRCLE_BASE_SIZE = 650
SMOOTH_ALPHA = 0.6
SENSITIVITY_DEFAULT = 1.0
COLORMAP = "plasma"

DEFAULT_ZONES = {
    "ankle_L": {"pos": [0.48, 0.09], "band": [20, 120], "label": "Ankle L", "gain": 1.0},
    "ankle_R": {"pos": [0.55, 0.09], "band": [20, 120], "label": "Ankle R", "gain": 1.0},
    "hip_L":   {"pos": [0.4303, 0.296], "band": [60, 250], "label": "Hip L", "gain": 1.0},
    "hip_R":   {"pos": [0.572, 0.296], "band": [60, 250], "label": "Hip R", "gain": 1.0},
    "chest_L": {"pos": [0.46, 0.54], "band": [100, 700], "label": "Chest L", "gain": 1.0},
    "chest_R": {"pos": [0.54, 0.54], "band": [100, 700], "label": "Chest R", "gain": 1.0},
    "arm_L":   {"pos": [0.38, 0.69], "band": [400, 2000], "label": "Arm L", "gain": 1.0},
    "arm_R":   {"pos": [0.62, 0.69], "band": [400, 2000], "label": "Arm R", "gain": 1.0},
    "hand_L":  {"pos": [0.22, 0.58], "band": [1500, 6000], "label": "Hand L", "gain": 1.0},
    "hand_R":  {"pos": [0.77, 0.58], "band": [1500, 6000], "label": "Hand R", "gain": 1.0},
}

# ---------------- Utilities ----------------
def ensure_file_exists(path, name="File"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")

def load_zones(path, fallback=DEFAULT_ZONES):
    if os.path.exists(path):
        with open(path, "r") as f:
            d = json.load(f)
        for k in d:
            if "gain" not in d[k]:
                d[k]["gain"] = 1.0
        return d
    else:
        with open(path, "w") as f:
            json.dump(fallback, f, indent=2)
        return json.loads(json.dumps(fallback))

def save_zones(path, zones):
    with open(path, "w") as f:
        json.dump(zones, f, indent=2)

def read_audio_any(path):
    """
    Read audio into (left, right, mono, sr).
    Prefer soundfile; fallback to pydub for mp3.
    Returns numpy float32 arrays in range [-1,1].
    """
    try:
        data, sr = sf.read(path, always_2d=True)
        # Convert to float32 between -1 and 1 if necessary
        if data.dtype.kind == 'i':
            # integer sample types -> normalize
            maxval = float(2 ** (8 * data.dtype.itemsize - 1))
            data = data.astype(np.float32) / maxval
        left = data[:,0].astype(np.float32)
        right = data[:,1].astype(np.float32) if data.shape[1] > 1 else left.copy()
        mono = 0.5*(left + right)
        return left, right, mono, sr
    except Exception:
        if not HAS_PYDUB:
            raise RuntimeError("Cannot read audio with soundfile and pydub not available. Install pydub or provide a wav.")
        audio = AudioSegment.from_file(path)
        sr = audio.frame_rate
        samples = np.array(audio.get_array_of_samples())
        # pydub returns interleaved samples for stereo
        samples = samples.astype(np.float32)
        if audio.channels == 1:
            left = samples / (2**(8*audio.sample_width - 1))
            right = left.copy()
        else:
            left = samples[0::audio.channels] / (2**(8*audio.sample_width - 1))
            right = samples[1::audio.channels] / (2**(8*audio.sample_width - 1))
        mono = 0.5*(left + right)
        return left.astype(np.float32), right.astype(np.float32), mono.astype(np.float32), sr

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
    for i in range(1,len(arr)):
        sm[i] = (1-a)*sm[i-1] + a*arr[i]
    return sm

def play_buffer_simpleaudio(mono, sr):
    """
    Play mono float32 [-1,1] using simpleaudio non-blocking and return PlayObject.
    """
    if not HAS_SIMPLEAUDIO:
        return None
    # convert to int16
    arr16 = np.int16(np.clip(mono, -1.0, 1.0) * 32767)
    try:
        play_obj = sa.play_buffer(arr16.tobytes(), 1, 2, sr)
        return play_obj
    except Exception:
        return None

# ---------------- Main visualizer ----------------
class HapticVisualizer:
    def __init__(self, audio_path=AUDIO_FILE, zones_json=ZONES_JSON):
        ensure_file_exists(audio_path, "Audio")
        self.audio_path = audio_path
        self.zones_json = zones_json
        self.zones = load_zones(zones_json)

        # load audio
        self.left, self.right, self.mono, self.sr = read_audio_any(audio_path)
        
        pygame.mixer.init(frequency=self.sr)
        pygame.mixer.music.load(self.audio_path)
        pygame.mixer.music.play()

        # compute STFT magnitudes
        self.freqs, self.times, self.S_left = stft_abs(self.left, self.sr)
        _, _, self.S_right = stft_abs(self.right, self.sr)

        self.frame_count = self.S_left.shape[1]
        self.frame_rate = self.sr / HOP_LENGTH
        self.duration = len(self.mono) / float(self.sr)

        # UI params
        self.smoothing = 0.6
        self.sensitivity = SENSITIVITY_DEFAULT
        self.preset = "normal"  # normal/bass/treble

        # precompute energies and normalized
        self.energies = self._compute_zone_energies()
        self.normalized = self._normalize_and_smooth(self.energies, self.smoothing)

        # matplotlib figure
        plt.rcParams['figure.facecolor'] = FIG_BG
        self.fig = plt.figure(figsize=(14,8), facecolor=FIG_BG)
        gs = self.fig.add_gridspec(1, 5, width_ratios=[3,0,0,0,2], wspace=0.04)
        self.ax_body = self.fig.add_subplot(gs[0,0])
        self.ax_body.set_facecolor("white")
        self.ax_body.set_xlim(0,1); self.ax_body.set_ylim(0,1)
        self.ax_body.set_xticks([]); self.ax_body.set_yticks([])
        self.ax_body.set_title("Haptic Body Map — click a marker to analyze", color="black", fontsize=14)

        self.ax_control = self.fig.add_subplot(gs[:,4]); self.ax_control.axis("off")

        self._draw_body()
        self.markers = {}
        self._create_markers()

        self.cmap = plt.get_cmap(COLORMAP)
        self.norm = plt.Normalize(0,1)

        # interaction state
        self._dragging = {"zone": None}
        self._press_candidate = {"zone": None, "pos": None}
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

        # controls & preset explanation
        self._build_controls()

        # start playback and animation
        self.play_obj = None
        self.state = {"playing": False, "start_time": None}
        self._start_playback()   # autoplay as requested

        from matplotlib.animation import FuncAnimation
        self.anim = FuncAnimation(self.fig, self._update_frame, frames=self.frame_count, interval=1000/self.frame_rate, blit=False)

    # ------------- draw body -------------
    def _draw_body(self):
        ax = self.ax_body
        torso_pts = [(0.48,0.72),(0.52,0.72),(0.62,0.45),(0.57,0.30),(0.50,0.26),(0.43,0.30),(0.38,0.45)]
        torso = Polygon(torso_pts, closed=True, facecolor=BODY_FILL, edgecolor=BODY_EDGE, linewidth=1.2, zorder=1)
        ax.add_patch(torso)
        head = Ellipse((0.50,0.88), 0.14, 0.16, facecolor=BODY_FILL, edgecolor=BODY_EDGE, zorder=1)
        ax.add_patch(head)
        left_arm = Polygon([(0.38,0.70),(0.22,0.58),(0.26,0.55),(0.40,0.66)], closed=True, facecolor=BODY_FILL, edgecolor=BODY_EDGE, zorder=1)
        right_arm = Polygon([(0.62,0.70),(0.78,0.58),(0.74,0.55),(0.60,0.66)], closed=True, facecolor=BODY_FILL, edgecolor=BODY_EDGE, zorder=1)
        ax.add_patch(left_arm); ax.add_patch(right_arm)
        left_leg = Polygon([(0.47,0.26),(0.46,0.05),(0.50,0.05),(0.51,0.26)], closed=True, facecolor=BODY_FILL, edgecolor=BODY_EDGE, zorder=1)
        right_leg = Polygon([(0.51,0.26),(0.54,0.05),(0.58,0.05),(0.55,0.26)], closed=True, facecolor=BODY_FILL, edgecolor=BODY_EDGE, zorder=1)
        ax.add_patch(left_leg); ax.add_patch(right_leg)

    # ------------- markers -------------
    def _create_markers(self):
        for z, info in self.zones.items():
            x,y = info["pos"]
            color = (0.45,0.35,0.75,0.85) if z.endswith("_L") else (0.30,0.60,0.9,0.85)
            sc = self.ax_body.scatter([x],[y], s=CIRCLE_BASE_SIZE, facecolors=[color], edgecolors='k', linewidths=0.6, zorder=10)
            ann = self.ax_body.text(x, y+0.03, info.get("label", z), ha='center', fontsize=9, zorder=11)
            self.markers[z] = {"scatter": sc, "ann": ann}
        self.fig.canvas.draw_idle()

    # ------------- energies -------------
    def _compute_zone_energies(self):
        energies = {}
        for z, info in self.zones.items():
            low, high = info["band"]
            if z.endswith("_L"):
                energies[z] = band_energy_mean(self.S_left, self.freqs, low, high)
            elif z.endswith("_R"):
                energies[z] = band_energy_mean(self.S_right, self.freqs, low, high)
            else:
                energies[z] = 0.5*(band_energy_mean(self.S_left, self.freqs, low, high) + band_energy_mean(self.S_right, self.freqs, low, high))
        return energies

    def _normalize_and_smooth(self, energies, smoothing):
        gmax = max((np.max(v) if v.size else 0.0) for v in energies.values()) or 1.0
        out = {}
        for k, arr in energies.items():
            n = arr / gmax
            out[k] = smooth_arr(n, alpha=smoothing)
        return out

    # ------------- playback -------------
    def _start_playback(self):
        # Attempt to play using simpleaudio. If unavailable, warn but still animate.
        def playback_thread():
            if HAS_SIMPLEAUDIO:
                try:
                    self.play_obj = play_buffer_simpleaudio(self.mono, self.sr)
                    if self.play_obj is not None:
                        self.state["playing"] = True
                        self.state["start_time"] = time.time()
                        self._set_status("Playing (simpleaudio)")
                        return
                except Exception as e:
                    print("simpleaudio playback failed:", e)
            # fallback: write tmp wav and try to open with simpleaudio (again) or just start animation timing
            try:
                tmp = os.path.join(tempfile.gettempdir(), "__haptic_tmp.wav")
                sf.write(tmp, self.mono, self.sr)
                if HAS_SIMPLEAUDIO:
                    data, sr2 = sf.read(tmp, always_2d=False)
                    if data.dtype.kind == 'f':
                        arr16 = np.int16(np.clip(data, -1.0, 1.0)*32767)
                    else:
                        arr16 = np.int16(data)
                    self.play_obj = sa.play_buffer(arr16.tobytes(), 1 if data.ndim==1 else data.shape[1], 2, sr2)
                    self.state["playing"] = True
                    self.state["start_time"] = time.time()
                    self._set_status("Playing (fallback)")
                    return
            except Exception as e:
                print("Fallback playback failed:", e)
            # if no playback, still start animation timing (but user won't hear music)
            self.state["playing"] = True
            self.state["start_time"] = time.time()
            self._set_status("Playing (no audio library)")

        t = threading.Thread(target=playback_thread, daemon=True)
        t.start()
        # short sleep to let playback thread set start_time before animation begins
        time.sleep(0.05)

    # ------------- controls & explanation -------------
    def _build_controls(self):
        # Preset buttons
        ax_normal = self.fig.add_axes([0.72, 0.92, 0.22, 0.04])
        b_normal = Button(ax_normal, "Normal")
        b_normal.on_clicked(self._set_normal)
        ax_bass = self.fig.add_axes([0.72, 0.86, 0.22, 0.04])
        b_bass = Button(ax_bass, "Bass-Heavy")
        b_bass.on_clicked(self._set_bass)
        ax_treble = self.fig.add_axes([0.72, 0.80, 0.22, 0.04])
        b_treble = Button(ax_treble, "Treble-Heavy")
        b_treble.on_clicked(self._set_treble)

        # explanation text box (static area under presets)
        expl_ax = self.fig.add_axes([0.72, 0.67, 0.22, 0.12])
        expl_ax.axis("off")
        self.expl_text = expl_ax.text(0, 1, "Preset explanation will appear here.", va='top', ha='left', color='white', fontsize=9, wrap=True)

        # sliders
        ax_smooth = self.fig.add_axes([0.72, 0.58, 0.22, 0.03])
        self.s_smooth = Slider(ax_smooth, "Smoothing", 0.0, 0.95, valinit=self.smoothing)
        self.s_smooth.on_changed(self._on_smooth)
        ax_sens = self.fig.add_axes([0.72, 0.53, 0.22, 0.03])
        self.s_sens = Slider(ax_sens, "Sensitivity", 0.2, 3.0, valinit=self.sensitivity)
        self.s_sens.on_changed(self._on_sens)

        # info/status text (minimal — updated on events only)
        self.status_text = self.fig.text(0.72, 0.45, "", color="white", fontsize=9)
        self._update_preset_explanation()  # fill initial explanation

    def _set_normal(self, event=None):
        self.preset = "normal"
        self._update_preset_explanation()
        self._set_status("Preset: Normal")

    def _set_bass(self, event=None):
        self.preset = "bass"
        self._update_preset_explanation()
        self._set_status("Preset: Bass-Heavy")

    def _set_treble(self, event=None):
        self.preset = "treble"
        self._update_preset_explanation()
        self._set_status("Preset: Treble-Heavy")

    def _update_preset_explanation(self):
        if self.preset == "normal":
            txt = ("Normal: frequency bands are mapped evenly across body regions. "
                   "Bass to legs/hips, mids to torso, highs to arms/hands.")
        elif self.preset == "bass":
            txt = ("Bass-Heavy: low frequencies are amplified for leg/hip zones. "
                   "Good for emphasizing kick and basslines.")
        else:
            txt = ("Treble-Heavy: high frequencies are amplified for arm/hand zones. "
                   "Good for melodies and bright timbres.")
        self.expl_text.set_text(txt)
        self.fig.canvas.draw_idle()

    def _on_smooth(self, v):
        self.smoothing = float(v)
        self.normalized = self._normalize_and_smooth(self.energies, self.smoothing)
        self._set_status(f"Smoothing {self.smoothing:.2f}")

    def _on_sens(self, v):
        self.sensitivity = float(v)
        self._set_status(f"Sensitivity {self.sensitivity:.2f}")

    def _set_status(self, s):
        ts = datetime.now().strftime("%H:%M:%S")
        self.status_text.set_text(f"[{ts}] {s}")
        self.fig.canvas.draw_idle()

    # ------------- mouse interactions: dragging + click detection -------------
    def _on_press(self, event):
        if event.inaxes != self.ax_body:
            return
        x,y = event.xdata, event.ydata
        best, bestd = None, 1e9
        for z, it in self.markers.items():
            ox, oy = it["scatter"].get_offsets()[0]
            d = math.hypot(x-ox, y-oy)
            if d < bestd:
                bestd, best = d, z
        if best is not None and bestd < 0.12:
            # candidate for click or drag
            self._press_candidate["zone"] = best
            self._press_candidate["pos"] = (x,y)
            # do not start drag until movement passes threshold
        else:
            self._press_candidate["zone"] = None
            self._press_candidate["pos"] = None

    def _on_move(self, event):
        if self._press_candidate["zone"] is None:
            return
        if event.inaxes != self.ax_body:
            return
        x,y = event.xdata, event.ydata
        zx, zy = self._press_candidate["pos"]
        dx, dy = abs(x - zx), abs(y - zy)
        # if movement surpasses threshold, treat as drag start
        if dx > 0.01 or dy > 0.01:
            z = self._press_candidate["zone"]
            self._dragging["zone"] = z
            self._press_candidate["zone"] = None
            self._press_candidate["pos"] = None
            # fall-through to dragging update
        if self._dragging["zone"] is not None:
            z = self._dragging["zone"]
            x = max(0.02, min(0.98, event.xdata))
            y = max(0.02, min(0.98, event.ydata))
            self.zones[z]["pos"] = [x,y]
            sc = self.markers[z]["scatter"]; sc.set_offsets([[x,y]])
            self.markers[z]["ann"].set_position((x, y+0.03))
            self.fig.canvas.draw_idle()

    def _on_release(self, event):
        # if candidate exists and no drag started -> treat as click
        if self._press_candidate["zone"] is not None:
            z = self._press_candidate["zone"]
            self._press_candidate["zone"] = None
            self._press_candidate["pos"] = None
            # click action: open analytics
            self._open_zone_analytics(z)
            return
        # if dragging ended
        if self._dragging["zone"] is not None:
            z = self._dragging["zone"]
            save_zones(self.zones_json, self.zones)
            self._dragging["zone"] = None
            self._set_status(f"Placed {z} at {tuple(self.zones[z]['pos'])}")

    # ------------- analytics for a zone (3 plots) -------------
    def _open_zone_analytics(self, zone):
        # Prepare left/right band energies for the zone
        low, high = self.zones[zone]["band"]
        left_en = band_energy_mean(self.S_left, self.freqs, low, high)
        right_en = band_energy_mean(self.S_right, self.freqs, low, high)
        # normalize both to combined max for clarity
        combined_max = max(np.max(left_en), np.max(right_en), 1e-9)
        left_norm = left_en / combined_max
        right_norm = right_en / combined_max
        t = self.times

        fig, axes = plt.subplots(3,1, figsize=(8,8))
        fig.suptitle(f"Zone Analytics — {zone}", fontsize=12)

        # Plot 1: Left vs Right band energy over time
        axes[0].plot(t, left_norm, label="Left", color='tab:blue')
        axes[0].plot(t, right_norm, label="Right", color='tab:orange', alpha=0.9)
        axes[0].set_title("1) Left vs Right Band Energy Over Time")
        axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Normalized energy")
        axes[0].legend(); axes[0].grid(True)

        # Plot 2: Activation histogram (combined)
        combined = 0.5*(left_norm + right_norm)
        axes[1].hist(combined, bins=50, color='tab:green', edgecolor='black', linewidth=0.3)
        m,s,p = float(np.mean(combined)), float(np.std(combined)), float(np.max(combined))
        axes[1].set_title("2) Activation Distribution (combined L+R)")
        axes[1].set_xlabel("Normalized energy"); axes[1].set_ylabel("Count")
        axes[1].text(0.02, 0.85, f"Mean: {m:.4f}\nStd: {s:.4f}\nPeak: {p:.4f}", transform=axes[1].transAxes, bbox=dict(facecolor='white', alpha=0.7))

        # Plot 3: Spectrogram (mono)
        axes[2].set_title("3) Spectrogram (mono input)")
        try:
            axes[2].specgram(self.mono, NFFT=1024, Fs=self.sr, noverlap=512, cmap='magma')
            axes[2].set_xlabel("Time (s)"); axes[2].set_ylabel("Frequency (Hz)")
        except Exception:
            axes[2].text(0.1, 0.5, "Spectrogram unavailable", color='red')
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.show()

    # ------------- per-frame update (animation) -------------
    def _update_frame(self, frame_idx):
        # determine time index using playback time if available to keep sync
        idx = frame_idx
        if self.state["playing"] and self.state.get("start_time") is not None:
            t_elapsed = time.time() - self.state["start_time"]
            # compute approximate STFT frame index
            idx = int(t_elapsed * self.frame_rate)
        idx = min(max(0, idx), self.frame_count - 1)

        # compute values per zone applying preset bias and sensitivity
        vals = {}
        for z, arr in self.normalized.items():
            v = float(arr[idx]) if arr.size else 0.0
            bias = 1.0
            if self.preset == "bass":
                if ("ankle" in z) or ("hip" in z):
                    bias = 1.6
            elif self.preset == "treble":
                if ("hand" in z) or ("arm" in z):
                    bias = 1.6
            v = v * bias * float(self.zones[z].get("gain", 1.0)) * self.sensitivity
            v = max(0.0, min(1.0, v))
            vals[z] = v

        # update markers: color/alpha/size
        for z, item in self.markers.items():
            sc = item["scatter"]
            v = vals.get(z, 0.0)
            rgba = self.cmap(self.norm(v))
            sc.set_facecolor(rgba)
            sc.set_alpha(0.35 + 0.6 * v)
            sc.set_sizes([CIRCLE_BASE_SIZE * (0.5 + 1.0 * v)])
        # do NOT update verbose info each frame (user asked to avoid moving numbers)
        # only keep minimal status updated by events
        self.fig.canvas.draw_idle()
        return []

    # ------------- init / helpers -------------
    def show(self):
        plt.show()

# ---------------- main ----------------
def main(audio=AUDIO_FILE, zones_json=ZONES_JSON):
    if not os.path.exists(audio):
        print(f"Audio file not found: {audio}")
        return
    print("Starting Haptic Visualizer — autoplay. Make sure 'simpleaudio' is installed for playback.")
    vis = HapticVisualizer(audio, zones_json)
    vis.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Haptic Visualizer (autoplay, analytics)")
    parser.add_argument("--audio", default=AUDIO_FILE, help="Path to audio (mp3/wav)")
    parser.add_argument("--zones", default=ZONES_JSON, help="Zones JSON file")
    args = parser.parse_args()
    main(audio=args.audio, zones_json=args.zones)