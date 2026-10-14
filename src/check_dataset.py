from pathlib import Path

# Location of the RAVDESS dataset
dataset_path = Path("dataset/RAVDESS")

# Check whether the dataset folder exists
if not dataset_path.exists():
    print("ERROR: RAVDESS dataset folder was not found.")
    print(f"Expected location: {dataset_path.resolve()}")
    exit()

# Find all WAV audio files
wav_files = list(dataset_path.rglob("*.wav"))

print("RAVDESS dataset found!")
print(f"Dataset location: {dataset_path.resolve()}")
print(f"Total WAV files: {len(wav_files)}")

# Display the first 5 audio files
print("\nFirst 5 audio files:")

for file in wav_files[:5]:
    print(file)