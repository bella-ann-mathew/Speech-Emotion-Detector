from pathlib import Path
import librosa
import soundfile as sf
import numpy as np


# --------------------------------------------------
# Dataset locations
# --------------------------------------------------

dataset_path = Path("dataset/RAVDESS")
augmented_path = Path("dataset/RAVDESS_Augmented")

augmented_path.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Training actors only
# Actors 01-20
# --------------------------------------------------

train_actors = [
    f"Actor_{i:02d}"
    for i in range(1, 21)
]


# --------------------------------------------------
# Find training audio files
# --------------------------------------------------

wav_files = []

for actor in train_actors:

    actor_path = (
    dataset_path
    / "Audio_Speech_Actors_01-24_16k"
    / actor
)

    if actor_path.exists():
        wav_files.extend(
            actor_path.glob("*.wav")
        )


print("Audio Augmentation")
print("------------------")
print(f"Original training files: {len(wav_files)}")


# --------------------------------------------------
# Add noise
# --------------------------------------------------

for index, file in enumerate(wav_files):

    try:

        audio, sample_rate = librosa.load(
            file,
            sr=16000
        )

        # Calculate noise level
        noise = np.random.normal(
            0,
            0.005,
            len(audio)
        )

        # Add noise to original audio
        augmented_audio = audio + noise

        # Keep values within valid audio range
        augmented_audio = np.clip(
            augmented_audio,
            -1.0,
            1.0
        )

        # Save augmented file
        output_file = (
            augmented_path
            / f"{file.parent.name}_{file.stem}_noise.wav"
        )

        sf.write(
            output_file,
            augmented_audio,
            sample_rate
        )

        if (index + 1) % 100 == 0:

            print(
                f"Processed {index + 1}/{len(wav_files)} files"
            )

    except Exception as e:

        print(
            f"Error processing {file}: {e}"
        )


print("\nAugmentation completed!")

print(
    f"Augmented files saved: {len(wav_files)}"
)

print(
    f"Location: {augmented_path}"
)