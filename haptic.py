import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pygame
import time

# === USER SETTINGS ===
audio_path = "test.mp3"  # Replace with your audio file path
hop_length = 1024
n_fft = 2048

# === LOAD AUDIO ===
y, sr = librosa.load(audio_path, sr=None, mono=True)
S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
freqs = librosa.fft_frequencies(sr=sr)
times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)

# === DEFINE BODY ZONES AND FREQUENCY BANDS ===
zones = {
    "Ankles": (20, 100),
    "Chest": (100, 400),
    "Arms": (400, 1500),
    "Hands": (1500, 4000),
}

def band_energy(low, high):
    idx = np.where((freqs >= low) & (freqs < high))[0]
    return np.mean(S[idx, :], axis=0)

energies = {zone: band_energy(low, high) for zone, (low, high) in zones.items()}

# Normalize energy for consistent color scaling
max_val = max([np.max(v) for v in energies.values()])
for z in energies:
    energies[z] /= max_val

# === PYGAME SOUND PLAYBACK ===
pygame.mixer.init(frequency=sr)
pygame.mixer.music.load(audio_path)

# === MATPLOTLIB SETUP ===
fig, ax = plt.subplots(figsize=(6, 8))
ax.set_xlim(0, 6)
ax.set_ylim(0, 8)
ax.axis("off")
fig.patch.set_facecolor("black")

# Circle positions for body zones
positions = {
    "Ankles": (3, 1),
    "Chest": (3, 4),
    "Arms": (2, 5.5),
    "Hands": (4, 5.5),
}

# Create circles and text labels
circles = {}
for z, (x, y) in positions.items():
    c = plt.Circle((x, y), 0.6, color="blue", alpha=0.2)
    ax.add_patch(c)
    ax.text(x, y, z, ha="center", va="center", color="white", fontsize=10)
    circles[z] = c

# Add colorbar (vibration intensity)
sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=0, vmax=1))
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Vibration Intensity", color="white")
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

# === ANIMATION FUNCTION ===
frame_rate = sr / hop_length
frame_count = len(times)
start_time = None

def update(frame):
    global start_time

    if frame == 0:
        pygame.mixer.music.play()
        start_time = time.time()

    # Sync to audio playback
    current_time = time.time() - start_time
    target_frame = int(current_time * frame_rate)
    if target_frame >= frame_count:
        pygame.mixer.music.stop()
        plt.close()
        return circles.values()

    for z, c in circles.items():
        energy = energies[z][target_frame]
        c.set_color(plt.cm.plasma(energy))
        c.set_alpha(0.3 + 0.7 * energy)  # blend color by intensity

    return circles.values()

ani = FuncAnimation(fig, update, frames=frame_count, interval=1000/frame_rate, blit=False)
plt.show()
