import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pygame
import cv2
import time

# ============================================================
# USER SETTINGS
# ============================================================
AUDIO_PATH = "test.mp3"
BODY_IMAGE_PATH = "human.png"  # transparent PNG outline of human figure
HOP_LENGTH = 1024
N_FFT = 2048

# ============================================================
# LOAD AUDIO (Stereo)
# ============================================================
y, sr = librosa.load(AUDIO_PATH, sr=None, mono=False)
if y.ndim == 1:  # mono fallback
    y = np.vstack([y, y])

# Compute STFT for each channel
S_left = np.abs(librosa.stft(y[0], n_fft=N_FFT, hop_length=HOP_LENGTH))
S_right = np.abs(librosa.stft(y[1], n_fft=N_FFT, hop_length=HOP_LENGTH))
freqs = librosa.fft_frequencies(sr=sr)
times = librosa.frames_to_time(np.arange(S_left.shape[1]), sr=sr, hop_length=HOP_LENGTH)

# ============================================================
# DEFINE BODY ZONES AND FREQUENCY BANDS
# ============================================================
bands = {
    "ankle": (20, 100),
    "chest": (100, 300),
    "arm": (300, 1000),
    "hand": (1000, 4000),
}

# Compute energy per band and channel
def band_energy(S, low, high):
    idx = np.where((freqs >= low) & (freqs < high))[0]
    return np.mean(S[idx, :], axis=0)

energies = {}
for zone, (low, high) in bands.items():
    energies[f"{zone}_L"] = band_energy(S_left, low, high)
    energies[f"{zone}_R"] = band_energy(S_right, low, high)

# Normalize energies
max_val = max([np.max(v) for v in energies.values()])
for k in energies:
    energies[k] = energies[k] / max_val

# ============================================================
# LOAD BODY IMAGE
# ============================================================
img = cv2.imread(BODY_IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (400, 800))

fig, ax = plt.subplots(figsize=(4, 8))
fig.patch.set_facecolor("black")
ax.axis("off")
ax.imshow(img)

# ============================================================
# DEFINE ZONE POSITIONS (x, y in image coords)
# ============================================================
positions = {
    "ankle_L": (120, 720),
    "ankle_R": (280, 720),
    "chest_L": (170, 400),
    "chest_R": (230, 400),
    "arm_L": (100, 300),
    "arm_R": (300, 300),
    "hand_L": (70, 220),
    "hand_R": (330, 220),
}

# Create scatter points for visualization
scatters = {zone: ax.scatter(x, y, s=800, color="black") for zone, (x, y) in positions.items()}

# Add colorbar
sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=0, vmax=1))
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Vibration Intensity", color="white")
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

# ============================================================
# AUDIO PLAYBACK WITH PYGAME
# ============================================================
pygame.mixer.init(frequency=sr)
pygame.mixer.music.load(AUDIO_PATH)

frame_rate = sr / HOP_LENGTH
frame_count = len(times)
start_time = None

# ============================================================
# ANIMATION UPDATE FUNCTION
# ============================================================
def update(frame):
    global start_time

    if frame == 0:
        pygame.mixer.music.play()
        start_time = time.time()

    current_time = time.time() - start_time
    target_frame = int(current_time * frame_rate)
    if target_frame >= frame_count:
        pygame.mixer.music.stop()
        plt.close()
        return []

    # Update scatter points based on intensity
    for zone, sc in scatters.items():
        energy = energies[zone][target_frame]
        color = plt.cm.plasma(energy)
        sc.set_color(color)
        sc.set_alpha(0.2 + 0.8 * energy)
    return scatters.values()

ani = FuncAnimation(fig, update, frames=frame_count, interval=1000/frame_rate, blit=False)
plt.show()
