from pathlib import Path
from collections import Counter

# RAVDESS dataset location
dataset_path = Path("dataset/RAVDESS")

# RAVDESS emotion codes
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Find all WAV files
wav_files = list(dataset_path.rglob("*.wav"))

# Store emotion labels
emotions = []

for file in wav_files:
    # Example filename:
    # 03-01-05-01-02-01-12.wav

    filename = file.stem
    parts = filename.split("-")

    # The third part is the emotion code
    emotion_code = parts[2]

    if emotion_code in emotion_map:
        emotions.append(emotion_map[emotion_code])

# Count each emotion
emotion_counts = Counter(emotions)

print("RAVDESS Emotion Check")
print("---------------------")
print(f"Total WAV files: {len(wav_files)}")
print(f"Total labeled files: {len(emotions)}")

print("\nEmotion distribution:")

for code, emotion in emotion_map.items():
    print(f"{code} - {emotion}: {emotion_counts[emotion]}")