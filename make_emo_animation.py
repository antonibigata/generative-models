import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import torch

emotion_values = {
    "happy": (0.85, 0.75),  # Joy/Happiness
    "angry": (-0.443, 0.908),  # Anger
    "surprised": (0.0, 0.85),  # Surprise
    "sad": (-0.85, -0.35),  # Sadness
    "neutral": (0.0, 0.0),  # Neutral
    "fear": (0.181, 0.949),  # Fear
    "disgusted": (-0.8, 0.5),  # Disgust
    "contempt": (0.307, 0.535),  # Contempt
    "calm": (0.65, -0.5),  # Calmness
    "excited": (0.9, 0.9),  # Excitement
    "bored": (-0.6, -0.9),  # Boredom
    "confused": (-0.3, 0.4),  # Confusion
    "anxious": (-0.85, 0.9),  # Anxiety
    "confident": (0.7, 0.4),  # Confidence
    "frustrated": (-0.8, 0.6),  # Frustration
    "amused": (0.7, 0.5),  # Amusement
    "proud": (0.8, 0.4),  # Pride
    "ashamed": (-0.8, -0.3),  # Shame
    "grateful": (0.7, 0.2),  # Gratitude
    "jealous": (-0.7, 0.5),  # Jealousy
    "hopeful": (0.7, 0.3),  # Hope
    "disappointed": (-0.7, -0.3),  # Disappointment
    "curious": (0.5, 0.5),  # Curiosity
    "overwhelmed": (-0.6, 0.8),  # Overwhelm
    # Add more emotions as needed
}


def create_emotion_list(emotion_states, total_frames, emotion_values, accentuate=False):
    if accentuate:
        accentuated_values = {
            k: (max(min(v[0] * 1.5, 1), -1), max(min(v[1] * 1.5, 1), -1)) for k, v in emotion_values.items()
        }
        accentuated_values["neutral"] = (0.0, 0.0)  # Keep neutral as is
        emotion_values = accentuated_values

    if len(emotion_states) == 1:
        v, a = emotion_values[emotion_states[0]]
        valence = [v] * (total_frames + 2)
        arousal = [a] * (total_frames + 2)
    else:
        frames_per_transition = total_frames // (len(emotion_states) - 1)

        valence = []
        arousal = []

        for i in range(len(emotion_states) - 1):
            start_v, start_a = emotion_values[emotion_states[i]]
            end_v, end_a = emotion_values[emotion_states[i + 1]]

            v_values = np.linspace(start_v, end_v, frames_per_transition)
            a_values = np.linspace(start_a, end_a, frames_per_transition)

            valence.extend(v_values)
            arousal.extend(a_values)

        valence = valence[:total_frames]
        arousal = arousal[:total_frames]
        # Save valence and arousal as numpy arrays
        valence = np.array(valence)
        arousal = np.array(arousal)

    return (torch.tensor(valence), torch.tensor(arousal), torch.zeros(total_frames))


emotion_states = ["surprised", "angry", "calm", "sad", "happy"]

valence, arousal, zero = create_emotion_list(emotion_states, 25 * 44, emotion_values, accentuate=True)

# Create the figure and axis with dark style
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
ax.set_facecolor("black")

# Add grid and axes
ax.grid(color="gray", linestyle="--", alpha=0.3)
ax.axhline(0, color="white", linewidth=0.8)  # Horizontal line
ax.axvline(0, color="white", linewidth=0.8)  # Vertical line
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

# Add gridlines and labels with white text
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_xlabel("Valence (Pleasant → Unpleasant)", color="white")
ax.set_ylabel("Arousal (Low → High)", color="white")
ax.tick_params(colors="white")

# Plot emotion points with white text
for emotion in emotion_states:
    v, a = emotion_values[emotion]
    ax.text(v, a, emotion, fontsize=8, ha="center", va="center", color="white")

# Animation function with red marker
(arrow,) = ax.plot([], [], "ro", markersize=10, markerfacecolor="red")  # Red animated marker


def init():
    return (arrow,)


def update(frame):
    x = valence[frame]
    y = arousal[frame]
    arrow.set_data(x, y)
    return (arrow,)


# Create the animation
ani = FuncAnimation(fig, update, frames=len(valence), init_func=init, blit=True)

# Save animation
writer = FFMpegWriter(fps=25)
ani.save("emotion_animation.mp4", writer=writer)

plt.show()
