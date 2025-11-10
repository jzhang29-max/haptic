"""
Enhanced Haptic Visualizer - Modern UI with improved controls and analytics

Key improvements:
- Cleaner, modern UI with better color scheme
- Working preset buttons with visual feedback
- Improved slider controls with better responsiveness
- Enhanced analytics plots with professional styling
- Smoother animations and better performance
- Clear visual hierarchy and spacing
"""

import os
import json
import math
import time
import tempfile
import threading
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, FancyBboxPatch
from matplotlib.widgets import Button, Slider
from scipy import signal
import soundfile as sf

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except Exception:
    HAS_PYDUB = False

try:
    import simpleaudio as sa
    HAS_SIMPLEAUDIO = True
except Exception:
    HAS_SIMPLEAUDIO = False

# ---------------- Enhanced Config ----------------
AUDIO_FILE = "test.mp3"
ZONES_JSON = "zones_user.json"

N_FFT = 2048
HOP_LENGTH = 1024

# Modern color scheme
BG_COLOR = "#1a1a2e"
PANEL_BG = "#16213e"
BODY_BG = "#f5f5f5"
BODY_FILL = "#ffffff"
BODY_EDGE = "#2d3436"
ACCENT_COLOR = "#4a90e2"
ACCENT_HOVER = "#357abd"

CIRCLE_BASE_SIZE = 650
SMOOTH_ALPHA = 0.6
SENSITIVITY_DEFAULT = 1.0
COLORMAP = "viridis"

DEFAULT_ZONES = {
    "ankle_L": {"pos": [0.465, 0.09], "band": [20, 120], "label": "Ankle L", "gain": 1.0},
    "ankle_R": {"pos": [0.535, 0.09], "band": [20, 120], "label": "Ankle R", "gain": 1.0},
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
    """Read audio into (left, right, mono, sr)."""
    try:
        data, sr = sf.read(path, always_2d=True)
        if data.dtype.kind == 'i':
            maxval = float(2 ** (8 * data.dtype.itemsize - 1))
            data = data.astype(np.float32) / maxval
        left = data[:,0].astype(np.float32)
        right = data[:,1].astype(np.float32) if data.shape[1] > 1 else left.copy()
        mono = 0.5*(left + right)
        return left, right, mono, sr
    except Exception:
        if not HAS_PYDUB:
            raise RuntimeError("Cannot read audio. Install pydub or provide a wav file.")
        audio = AudioSegment.from_file(path)
        sr = audio.frame_rate
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        if audio.channels == 1:
            left = samples / (2**(8*audio.sample_width - 1))
            right = left.copy()
        else:
            left = samples[0::audio.channels] / (2**(8*audio.sample_width - 1))
            right = samples[1::audio.channels] / (2**(8*audio.sample_width - 1))
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
    if arr.size == 0:
        return arr
    sm = np.zeros_like(arr)
    sm[0] = arr[0]
    for i in range(1, len(arr)):
        sm[i] = (1-alpha)*sm[i-1] + alpha*arr[i]
    return sm

def play_buffer_simpleaudio(mono, sr):
    """Play audio using simpleaudio."""
    if not HAS_SIMPLEAUDIO:
        return None
    arr16 = np.int16(np.clip(mono, -1.0, 1.0) * 32767)
    try:
        return sa.play_buffer(arr16.tobytes(), 1, 2, sr)
    except Exception:
        return None

# ---------------- Main Visualizer ----------------
class HapticVisualizer:
    def __init__(self, audio_path=AUDIO_FILE, zones_json=ZONES_JSON):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        self.audio_path = audio_path
        self.zones_json = zones_json
        self.zones = load_zones(zones_json)

        # Load audio
        print("Loading audio...")
        self.left, self.right, self.mono, self.sr = read_audio_any(audio_path)
        
        # Initialize pygame mixer for music playback
        import pygame
        pygame.mixer.init(frequency=self.sr)
        
        # Compute STFT
        print("Computing spectrograms...")
        self.freqs, self.times, self.S_left = stft_abs(self.left, self.sr)
        _, _, self.S_right = stft_abs(self.right, self.sr)

        self.frame_count = self.S_left.shape[1]
        self.frame_rate = self.sr / HOP_LENGTH
        self.duration = len(self.mono) / float(self.sr)

        # UI state
        self.smoothing = 0.6
        self.sensitivity = SENSITIVITY_DEFAULT
        self.preset = "normal"

        # Compute energies
        print("Processing zones...")
        self.energies = self._compute_zone_energies()
        self.normalized = self._normalize_and_smooth(self.energies, self.smoothing)

        # Create figure with modern styling
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 9), facecolor=BG_COLOR)
        
        # Create layout with better proportions
        gs = self.fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.03,
                                   left=0.03, right=0.97, top=0.94, bottom=0.05)
        
        # Body visualization panel
        self.ax_body = self.fig.add_subplot(gs[0, 0])
        self.ax_body.set_facecolor(BODY_BG)
        self.ax_body.set_xlim(0, 1)
        self.ax_body.set_ylim(0, 1)
        self.ax_body.set_xticks([])
        self.ax_body.set_yticks([])
        self.ax_body.spines['top'].set_color(BODY_EDGE)
        self.ax_body.spines['bottom'].set_color(BODY_EDGE)
        self.ax_body.spines['left'].set_color(BODY_EDGE)
        self.ax_body.spines['right'].set_color(BODY_EDGE)
        self.ax_body.set_title("Haptic Body Map — Click markers to analyze", 
                              color='white', fontsize=16, pad=15, fontweight='bold')

        # Control panel
        self.ax_control = self.fig.add_subplot(gs[0, 1])
        self.ax_control.set_facecolor(PANEL_BG)
        self.ax_control.axis("off")

        # Draw body and markers
        self._draw_body()
        self.markers = {}
        self._create_markers()

        # Setup colormap
        self.cmap = plt.get_cmap(COLORMAP)
        self.norm = plt.Normalize(0, 1)

        # Interaction state
        self._dragging = {"zone": None}
        self._press_candidate = {"zone": None, "pos": None}
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

        # Build controls
        self._build_controls()

        # Playback state
        self.play_obj = None
        self.state = {"playing": False, "start_time": None}
        
        # Start playback
        print("Starting playback...")
        self._start_playback()

        # Animation
        from matplotlib.animation import FuncAnimation
        self.anim = FuncAnimation(self.fig, self._update_frame, 
                                 frames=self.frame_count, 
                                 interval=1000/self.frame_rate, 
                                 blit=False)
        
        print("Visualizer ready!")

    def _draw_body(self):
        """Draw anatomical body silhouette."""
        ax = self.ax_body
        
        # Torso
        torso_pts = [(0.48, 0.72), (0.52, 0.72), (0.62, 0.45), 
                     (0.57, 0.30), (0.50, 0.26), (0.43, 0.30), (0.38, 0.45)]
        torso = Polygon(torso_pts, closed=True, facecolor=BODY_FILL, 
                       edgecolor=BODY_EDGE, linewidth=2, zorder=1)
        ax.add_patch(torso)
        
        # Head
        head = Ellipse((0.50, 0.88), 0.14, 0.16, facecolor=BODY_FILL, 
                      edgecolor=BODY_EDGE, linewidth=2, zorder=1)
        ax.add_patch(head)
        
        # Arms
        left_arm = Polygon([(0.38, 0.70), (0.22, 0.58), (0.26, 0.55), (0.40, 0.66)], 
                          closed=True, facecolor=BODY_FILL, edgecolor=BODY_EDGE, 
                          linewidth=2, zorder=1)
        right_arm = Polygon([(0.62, 0.70), (0.78, 0.58), (0.74, 0.55), (0.60, 0.66)], 
                           closed=True, facecolor=BODY_FILL, edgecolor=BODY_EDGE, 
                           linewidth=2, zorder=1)
        ax.add_patch(left_arm)
        ax.add_patch(right_arm)
        
        # Legs - centered at hip intersection
        left_leg = Polygon([(0.465, 0.26), (0.445, 0.05), (0.485, 0.05), (0.495, 0.26)], 
                          closed=True, facecolor=BODY_FILL, edgecolor=BODY_EDGE, 
                          linewidth=2, zorder=1)
        right_leg = Polygon([(0.505, 0.26), (0.515, 0.05), (0.555, 0.05), (0.535, 0.26)], 
                           closed=True, facecolor=BODY_FILL, edgecolor=BODY_EDGE, 
                           linewidth=2, zorder=1)
        ax.add_patch(left_leg)
        ax.add_patch(right_leg)

    def _create_markers(self):
        """Create haptic zone markers."""
        for z, info in self.zones.items():
            x, y = info["pos"]
            color = '#ff6b9d' if z.endswith("_L") else '#4a90e2'
            
            sc = self.ax_body.scatter([x], [y], s=CIRCLE_BASE_SIZE, 
                                     facecolors=[color], edgecolors='white', 
                                     linewidths=2, zorder=10, alpha=0.7)
            
            ann = self.ax_body.text(x, y + 0.03, info.get("label", z), 
                                   ha='center', fontsize=10, color='black',
                                   fontweight='bold', zorder=11)
            self.markers[z] = {"scatter": sc, "ann": ann}

    def _compute_zone_energies(self):
        """Compute frequency band energies for each zone."""
        energies = {}
        for z, info in self.zones.items():
            low, high = info["band"]
            if z.endswith("_L"):
                energies[z] = band_energy_mean(self.S_left, self.freqs, low, high)
            elif z.endswith("_R"):
                energies[z] = band_energy_mean(self.S_right, self.freqs, low, high)
            else:
                L = band_energy_mean(self.S_left, self.freqs, low, high)
                R = band_energy_mean(self.S_right, self.freqs, low, high)
                energies[z] = 0.5 * (L + R)
        return energies

    def _normalize_and_smooth(self, energies, smoothing):
        """Normalize and smooth energy values."""
        gmax = max((np.max(v) if v.size else 0.0) for v in energies.values()) or 1.0
        out = {}
        for k, arr in energies.items():
            n = arr / gmax
            out[k] = smooth_arr(n, alpha=smoothing)
        return out

    def _start_playback(self):
        """Start audio playback using pygame."""
        def playback_thread():
            try:
                import pygame
                pygame.mixer.music.load(self.audio_path)
                pygame.mixer.music.play()
                self.state["playing"] = True
                self.state["start_time"] = time.time()
                self._set_status("▶ Playing")
            except Exception as e:
                print(f"Playback error: {e}")
                self.state["playing"] = True
                self.state["start_time"] = time.time()
                self._set_status("▶ Playing (no audio)")

        t = threading.Thread(target=playback_thread, daemon=True)
        t.start()
        time.sleep(0.05)

    def _build_controls(self):
        """Build control panel UI."""
        # Title
        self.fig.text(0.73, 0.94, "CONTROL PANEL", color='white', 
                     fontsize=13, fontweight='bold')
        
        # Preset buttons with better styling
        btn_width, btn_height = 0.20, 0.04
        btn_x = 0.74
        
        # Normal preset
        ax_normal = self.fig.add_axes([btn_x, 0.88, btn_width, btn_height])
        self.btn_normal = Button(ax_normal, "Normal", 
                                color=ACCENT_COLOR, hovercolor=ACCENT_HOVER)
        self.btn_normal.label.set_color('white')
        self.btn_normal.label.set_fontweight('bold')
        self.btn_normal.label.set_fontsize(10)
        self.btn_normal.on_clicked(self._set_normal)
        
        # Bass preset
        ax_bass = self.fig.add_axes([btn_x, 0.83, btn_width, btn_height])
        self.btn_bass = Button(ax_bass, "Bass-Heavy", 
                              color='#2c3e50', hovercolor='#34495e')
        self.btn_bass.label.set_color('white')
        self.btn_bass.label.set_fontweight('bold')
        self.btn_bass.label.set_fontsize(10)
        self.btn_bass.on_clicked(self._set_bass)
        
        # Treble preset
        ax_treble = self.fig.add_axes([btn_x, 0.78, btn_width, btn_height])
        self.btn_treble = Button(ax_treble, "Treble-Heavy", 
                                color='#2c3e50', hovercolor='#34495e')
        self.btn_treble.label.set_color('white')
        self.btn_treble.label.set_fontweight('bold')
        self.btn_treble.label.set_fontsize(10)
        self.btn_treble.on_clicked(self._set_treble)
        
        # Explanation box - more compact
        self.fig.text(0.73, 0.74, "PRESET INFO", color='white', 
                     fontsize=10, fontweight='bold')
        
        expl_ax = self.fig.add_axes([0.73, 0.62, 0.24, 0.11])
        expl_ax.set_facecolor('#0f1419')
        expl_ax.axis("off")
        self.expl_text = expl_ax.text(0.05, 0.95, "", va='top', ha='left', 
                                      color='#cccccc', fontsize=8.5, wrap=True)
        
        # Sliders - better positioned
        self.fig.text(0.73, 0.58, "PARAMETERS", color='white', 
                     fontsize=10, fontweight='bold')
        
        ax_smooth = self.fig.add_axes([0.74, 0.53, 0.20, 0.025])
        self.s_smooth = Slider(ax_smooth, "Smoothing", 0.0, 0.95, 
                              valinit=self.smoothing, color=ACCENT_COLOR)
        self.s_smooth.label.set_color('white')
        self.s_smooth.label.set_fontsize(9)
        self.s_smooth.valtext.set_color('white')
        self.s_smooth.valtext.set_fontsize(9)
        self.s_smooth.on_changed(self._on_smooth)
        
        ax_sens = self.fig.add_axes([0.74, 0.48, 0.20, 0.025])
        self.s_sens = Slider(ax_sens, "Sensitivity", 0.2, 3.0, 
                            valinit=self.sensitivity, color=ACCENT_COLOR)
        self.s_sens.label.set_color('white')
        self.s_sens.label.set_fontsize(9)
        self.s_sens.valtext.set_color('white')
        self.s_sens.valtext.set_fontsize(9)
        self.s_sens.on_changed(self._on_sens)
        
        # Instructions - compact spacing
        self.fig.text(0.73, 0.42, "INSTRUCTIONS", color='white', 
                     fontsize=10, fontweight='bold')
        instructions = ("• Click markers for analytics\n• Drag to reposition\n• Adjust sliders for effect\n• Try different presets")
        self.fig.text(0.73, 0.38, instructions, color='#999999', 
                     fontsize=8.5, verticalalignment='top', linespacing=1.5)
        
        # Live frequency spectrum visualization
        self.fig.text(0.73, 0.28, "LIVE SPECTRUM", color='white', 
                     fontsize=10, fontweight='bold')
        
        ax_spectrum = self.fig.add_axes([0.73, 0.06, 0.24, 0.20])
        ax_spectrum.set_facecolor('#0f1419')
        ax_spectrum.set_xlim(0, 8000)
        ax_spectrum.set_ylim(0, 1)
        ax_spectrum.set_xlabel('Frequency (Hz)', color='white', fontsize=8)
        ax_spectrum.set_ylabel('Magnitude', color='white', fontsize=8)
        ax_spectrum.tick_params(colors='white', labelsize=7)
        for spine in ax_spectrum.spines.values():
            spine.set_color('white')
        
        # Create frequency bins for display
        self.spectrum_freqs = np.linspace(0, 8000, 100)
        self.spectrum_line, = ax_spectrum.plot(self.spectrum_freqs, 
                                               np.zeros(100), 
                                               color='#4a90e2', 
                                               linewidth=2)
        ax_spectrum.fill_between(self.spectrum_freqs, 0, 0, 
                                alpha=0.3, color='#4a90e2')
        self.spectrum_fill = ax_spectrum.collections[0]
        self.ax_spectrum = ax_spectrum
        
        self._update_preset_explanation()

    def _set_normal(self, event=None):
        """Set normal preset."""
        self.preset = "normal"
        self._update_button_colors()
        self._update_preset_explanation()
        # Force re-normalize with current smoothing
        self.normalized = self._normalize_and_smooth(self.energies, self.smoothing)

    def _set_bass(self, event=None):
        """Set bass-heavy preset."""
        self.preset = "bass"
        self._update_button_colors()
        self._update_preset_explanation()
        self.normalized = self._normalize_and_smooth(self.energies, self.smoothing)

    def _set_treble(self, event=None):
        """Set treble-heavy preset."""
        self.preset = "treble"
        self._update_button_colors()
        self._update_preset_explanation()
        self.normalized = self._normalize_and_smooth(self.energies, self.smoothing)

    def _update_button_colors(self):
        """Update button colors to show active preset."""
        buttons = {
            'normal': self.btn_normal,
            'bass': self.btn_bass,
            'treble': self.btn_treble
        }
        
        for name, btn in buttons.items():
            if name == self.preset:
                btn.color = ACCENT_COLOR
                btn.hovercolor = ACCENT_HOVER
            else:
                btn.color = '#2c3e50'
                btn.hovercolor = '#34495e'
        
        self.fig.canvas.draw_idle()

    def _update_preset_explanation(self):
        """Update preset explanation text."""
        explanations = {
            "normal": ("Balanced frequency mapping.\n\nBass → Legs/Hips\nMids → Torso\nHighs → Arms/Hands"),
            "bass": ("Enhanced low-freq response for legs/hips.\n\nEmphasizes kick drums and basslines."),
            "treble": ("Amplified high-freq for arms/hands.\n\nHighlights melodies and vocals.")
        }
        self.expl_text.set_text(explanations.get(self.preset, ""))
        self.fig.canvas.draw_idle()

    def _on_smooth(self, v):
        """Handle smoothing slider change."""
        self.smoothing = float(v)
        self.normalized = self._normalize_and_smooth(self.energies, self.smoothing)

    def _on_sens(self, v):
        """Handle sensitivity slider change."""
        self.sensitivity = float(v)

    def _set_status(self, s):
        """Update status text (removed - no longer used)."""
        pass

    def _on_press(self, event):
        """Handle mouse press."""
        if event.inaxes != self.ax_body:
            return
        
        x, y = event.xdata, event.ydata
        best, bestd = None, 1e9
        
        for z, it in self.markers.items():
            ox, oy = it["scatter"].get_offsets()[0]
            d = math.hypot(x - ox, y - oy)
            if d < bestd:
                bestd, best = d, z
        
        if best is not None and bestd < 0.12:
            self._press_candidate["zone"] = best
            self._press_candidate["pos"] = (x, y)
        else:
            self._press_candidate["zone"] = None
            self._press_candidate["pos"] = None

    def _on_move(self, event):
        """Handle mouse move (dragging)."""
        if self._press_candidate["zone"] is None:
            return
        if event.inaxes != self.ax_body:
            return
        
        x, y = event.xdata, event.ydata
        zx, zy = self._press_candidate["pos"]
        dx, dy = abs(x - zx), abs(y - zy)
        
        if dx > 0.01 or dy > 0.01:
            z = self._press_candidate["zone"]
            self._dragging["zone"] = z
            self._press_candidate["zone"] = None
            self._press_candidate["pos"] = None
        
        if self._dragging["zone"] is not None:
            z = self._dragging["zone"]
            x = max(0.02, min(0.98, event.xdata))
            y = max(0.02, min(0.98, event.ydata))
            self.zones[z]["pos"] = [x, y]
            
            sc = self.markers[z]["scatter"]
            sc.set_offsets([[x, y]])
            self.markers[z]["ann"].set_position((x, y + 0.03))
            self.fig.canvas.draw_idle()

    def _on_release(self, event):
        """Handle mouse release."""
        if self._press_candidate["zone"] is not None:
            z = self._press_candidate["zone"]
            self._press_candidate["zone"] = None
            self._press_candidate["pos"] = None
            self._open_zone_analytics(z)
            return
        
        if self._dragging["zone"] is not None:
            z = self._dragging["zone"]
            save_zones(self.zones_json, self.zones)
            self._dragging["zone"] = None

    def _open_zone_analytics(self, zone):
        """Open analytics window for a zone."""
        low, high = self.zones[zone]["band"]
        
        # Determine if this is left or right zone
        is_left = zone.endswith("_L")
        is_right = zone.endswith("_R")
        
        if is_left:
            energy = band_energy_mean(self.S_left, self.freqs, low, high)
            side_label = "Left"
            side_color = '#ff6b9d'
        elif is_right:
            energy = band_energy_mean(self.S_right, self.freqs, low, high)
            side_label = "Right"
            side_color = '#4a90e2'
        else:
            # For center zones, use average
            left_en = band_energy_mean(self.S_left, self.freqs, low, high)
            right_en = band_energy_mean(self.S_right, self.freqs, low, high)
            energy = 0.5 * (left_en + right_en)
            side_label = "Center (L+R avg)"
            side_color = '#9b59b6'
        
        energy_max = max(np.max(energy), 1e-9)
        energy_norm = energy / energy_max
        t = self.times
        
        # Create styled analytics figure
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), facecolor='#1a1a2e')
        fig.suptitle(f"Zone Analytics — {zone}", fontsize=14, 
                    color='white', fontweight='bold')
        
        for ax in axes:
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
        
        # Plot 1: Energy over time for this side only
        axes[0].plot(t, energy_norm, label=side_label, color=side_color, linewidth=2)
        axes[0].set_title(f"{side_label} Band Energy Over Time ({low}-{high} Hz)", fontweight='bold')
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Normalized Energy")
        axes[0].legend(facecolor='#16213e', edgecolor='white', labelcolor='white')
        axes[0].grid(True, alpha=0.2, color='white')
        
        # Plot 2: Histogram
        axes[1].hist(energy_norm, bins=50, color=side_color, edgecolor='white', linewidth=0.5)
        m, s, p = np.mean(energy_norm), np.std(energy_norm), np.max(energy_norm)
        axes[1].set_title("Activation Distribution", fontweight='bold')
        axes[1].set_xlabel("Normalized Energy")
        axes[1].set_ylabel("Count")
        
        info_text = f"Mean: {m:.4f}\nStd: {s:.4f}\nPeak: {p:.4f}"
        axes[1].text(0.98, 0.95, info_text, transform=axes[1].transAxes,
                    ha='right', va='top', color='white', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='#0f1419', alpha=0.8))
        axes[1].grid(True, alpha=0.2, color='white')
        
        # Plot 3: Spectrogram
        axes[2].set_title("Spectrogram (Mono)", fontweight='bold')
        try:
            axes[2].specgram(self.mono, NFFT=1024, Fs=self.sr, 
                           noverlap=512, cmap='magma')
            axes[2].set_xlabel("Time (s)")
            axes[2].set_ylabel("Frequency (Hz)")
            axes[2].grid(True, alpha=0.2, color='white')
        except Exception as e:
            axes[2].text(0.5, 0.5, f"Spectrogram unavailable\n{str(e)}", 
                        color='#ff6b9d', ha='center', va='center',
                        transform=axes[2].transAxes)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    def _update_frame(self, frame_idx):
        """Update visualization for current frame."""
        # Sync with playback time
        idx = frame_idx
        if self.state["playing"] and self.state.get("start_time") is not None:
            t_elapsed = time.time() - self.state["start_time"]
            idx = int(t_elapsed * self.frame_rate)
        idx = min(max(0, idx), self.frame_count - 1)
        
        # Calculate values for each zone with preset bias
        vals = {}
        for z, arr in self.normalized.items():
            v = float(arr[idx]) if arr.size else 0.0
            
            # Apply preset bias - make it much more pronounced
            bias = 1.0
            if self.preset == "bass":
                if "ankle" in z or "hip" in z:
                    bias = 2.5  # Increased from 1.6
                else:
                    bias = 0.5  # Reduce others
            elif self.preset == "treble":
                if "hand" in z or "arm" in z:
                    bias = 2.5  # Increased from 1.6
                else:
                    bias = 0.5  # Reduce others
            
            # Apply gain and sensitivity
            v = v * bias * float(self.zones[z].get("gain", 1.0)) * self.sensitivity
            v = max(0.0, min(1.0, v))
            vals[z] = v
        
        # Update marker appearance
        for z, item in self.markers.items():
            sc = item["scatter"]
            v = vals.get(z, 0.0)
            
            # Color from colormap
            rgba = list(self.cmap(self.norm(v)))
            
            # Dynamic alpha based on intensity
            rgba[3] = 0.4 + 0.6 * v
            
            sc.set_facecolor([rgba])
            sc.set_sizes([CIRCLE_BASE_SIZE * (0.6 + 0.8 * v)])
        
        # Update live spectrum visualization
        if idx < self.frame_count:
            # Get current frame spectrum (average L+R)
            spectrum_slice = 0.5 * (self.S_left[:, idx] + self.S_right[:, idx])
            
            # Interpolate to display frequencies
            if len(self.freqs) > 0:
                spectrum_interp = np.interp(self.spectrum_freqs, self.freqs, spectrum_slice)
                spectrum_norm = spectrum_interp / (np.max(spectrum_interp) + 1e-9)
                
                self.spectrum_line.set_ydata(spectrum_norm)
                
                # Update fill
                self.spectrum_fill.remove()
                self.spectrum_fill = self.ax_spectrum.fill_between(
                    self.spectrum_freqs, 0, spectrum_norm,
                    alpha=0.3, color='#4a90e2'
                )
        
        return []

    def show(self):
        """Display the visualizer."""
        plt.show()


# ---------------- Main Entry Point ----------------
def main(audio=AUDIO_FILE, zones_json=ZONES_JSON):
    """Main function to run the visualizer."""
    if not os.path.exists(audio):
        print(f"Error: Audio file not found: {audio}")
        print("Please provide a valid audio file (MP3 or WAV)")
        return
    
    print("=" * 60)
    print("HAPTIC VISUALIZER - Enhanced Edition")
    print("=" * 60)
    print("\nFeatures:")
    print("  ✓ Real-time audio visualization")
    print("  ✓ Interactive zone analytics")
    print("  ✓ Draggable haptic markers")
    print("  ✓ Multiple presets (Normal, Bass, Treble)")
    print("  ✓ Adjustable smoothing and sensitivity")
    print("\nControls:")
    print("  • Click any marker to view detailed analytics")
    print("  • Drag markers to reposition them")
    print("  • Use sliders to adjust parameters")
    print("  • Switch presets to emphasize different frequencies")
    print("\n" + "=" * 60 + "\n")
    
    try:
        vis = HapticVisualizer(audio, zones_json)
        vis.show()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Enhanced Haptic Visualizer with analytics and presets"
    )
    parser.add_argument("--audio", default=AUDIO_FILE, 
                       help="Path to audio file (MP3/WAV)")
    parser.add_argument("--zones", default=ZONES_JSON, 
                       help="Path to zones configuration JSON")
    args = parser.parse_args()
    main(audio=args.audio, zones_json=args.zones)