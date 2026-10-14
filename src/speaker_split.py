from pathlib import Path
from collections import Counter

# RAVDESS dataset location
dataset_path = Path("dataset/RAVDESS")

# Find all WAV files
wav_files = list(dataset_path.rglob("*.wav"))

# Use actors 01-20 for training
# Use actors 21-24 for testing
train_actors = [f"Actor_{i:02d}" for i in range(1, 21)]
test_actors = [f"Actor_{i:02d}" for i in range(21, 25)]

train_files = []
test_files = []

for file in wav_files:
    actor_name = file.parent.name

    if actor_name in train_actors:
        train_files.append(file)

    elif actor_name in test_actors:
        test_files.append(file)


print("Speaker-Independent Dataset Split")
print("----------------------------------")

print(f"Total audio files: {len(wav_files)}")
print(f"Training files: {len(train_files)}")
print(f"Testing files: {len(test_files)}")

print("\nTraining actors:")
print(", ".join(train_actors))

print("\nTesting actors:")
print(", ".join(test_actors))

print("\nTraining emotion distribution:")

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

train_emotions = []

for file in train_files:
    parts = file.stem.split("-")
    emotion_code = parts[2]

    if emotion_code in emotion_map:
        train_emotions.append(emotion_map[emotion_code])

train_counts = Counter(train_emotions)

for emotion, count in train_counts.items():
    print(f"{emotion}: {count}")

print("\nTesting emotion distribution:")

test_emotions = []

for file in test_files:
    parts = file.stem.split("-")
    emotion_code = parts[2]

    if emotion_code in emotion_map:
        test_emotions.append(emotion_map[emotion_code])

test_counts = Counter(test_emotions)

for emotion, count in test_counts.items():
    print(f"{emotion}: {count}")