from pathlib import Path
import librosa
import numpy as np
import pickle


# --------------------------------------------------
# Dataset locations
# --------------------------------------------------

dataset_path = Path("dataset/RAVDESS")
augmented_path = Path("dataset/RAVDESS_Augmented")


# --------------------------------------------------
# Emotion mapping
# --------------------------------------------------

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


# --------------------------------------------------
# Training actors
# Actors 01-20 only
# --------------------------------------------------

train_actors = [
    f"Actor_{i:02d}"
    for i in range(1, 21)
]


# --------------------------------------------------
# Find augmented audio files
# --------------------------------------------------

wav_files = list(
    augmented_path.rglob("*.wav")
)


print("Augmented MFCC Feature Extraction")
print("----------------------------------")
print(f"Augmented audio files found: {len(wav_files)}")


# --------------------------------------------------
# Extract MFCC features
# --------------------------------------------------

features = []
labels = []


for index, file in enumerate(wav_files):

    try:

        audio, sample_rate = librosa.load(
            file,
            sr=16000
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )


        # --------------------------------------------------
        # Make all MFCCs the same size
        # --------------------------------------------------

        if mfcc.shape[1] < 130:

            pad_width = 130 - mfcc.shape[1]

            mfcc = np.pad(
                mfcc,
                ((0, 0), (0, pad_width)),
                mode="constant"
            )

        else:

            mfcc = mfcc[:, :130]


        # --------------------------------------------------
        # Add CNN channel dimension
        # --------------------------------------------------

        mfcc = mfcc[..., np.newaxis]


        # --------------------------------------------------
        # Get emotion from original filename
        #
        # Example:
        # Actor_01_03-01-05-01-01-01-01_noise.wav
        # --------------------------------------------------

        original_name = file.stem.replace(
            "_noise",
            ""
        )

        parts = original_name.split("_")

        original_filename = "_".join(
            parts[2:]
        )

        emotion_parts = original_filename.split("-")

        emotion_code = emotion_parts[2]

        emotion = emotion_map[emotion_code]


        # --------------------------------------------------
        # Store feature and label
        # --------------------------------------------------

        features.append(mfcc)
        labels.append(emotion)


        if (index + 1) % 100 == 0:

            print(
                f"Processed {index + 1}/{len(wav_files)} files"
            )


    except Exception as e:

        print(
            f"Error processing {file}: {e}"
        )


# --------------------------------------------------
# Convert to NumPy arrays
# --------------------------------------------------

features = np.array(features)
labels = np.array(labels)


# --------------------------------------------------
# Save augmented features
# --------------------------------------------------

output_file = Path(
    "dataset/mfcc_augmented_features.pkl"
)


with open(output_file, "wb") as file:

    pickle.dump(
        {
            "features": features,
            "labels": labels
        },
        file
    )


# --------------------------------------------------
# Final information
# --------------------------------------------------

print("\nMFCC extraction completed!")
print(f"Total augmented features: {len(features)}")
print(f"Feature shape: {features.shape}")
print(f"Labels: {len(labels)}")
print(f"Saved to: {output_file}")