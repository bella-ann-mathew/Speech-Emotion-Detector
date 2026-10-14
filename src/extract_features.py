from pathlib import Path
import librosa
import numpy as np
import pickle


# --------------------------------------------------
# Dataset location
# --------------------------------------------------

dataset_path = Path("dataset/RAVDESS")


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
# Find all audio files
# --------------------------------------------------

wav_files = sorted(
    dataset_path.rglob("*.wav")
)


print("MFCC Feature Extraction")
print("-----------------------")
print(f"Total audio files found: {len(wav_files)}")

print("\nStarting MFCC extraction...\n")


# --------------------------------------------------
# Lists for storing data
# --------------------------------------------------

features = []
labels = []
actors = []
file_names = []


# --------------------------------------------------
# Extract MFCC features
# --------------------------------------------------

for index, file in enumerate(wav_files):

    try:

        # Load audio
        audio, sample_rate = librosa.load(
            file,
            sr=16000
        )


        # Extract 40 MFCC coefficients
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )


        # --------------------------------------------------
        # Make every MFCC the same size
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
        # Add channel dimension for CNN
        # --------------------------------------------------

        mfcc = mfcc[..., np.newaxis]


        # --------------------------------------------------
        # Extract emotion from filename
        # --------------------------------------------------

        filename = file.stem

        parts = filename.split("-")

        emotion_code = parts[2]

        emotion = emotion_map[
            emotion_code
        ]


        # --------------------------------------------------
        # Extract actor name
        # --------------------------------------------------

        actor_name = file.parent.name


        # --------------------------------------------------
        # Store information
        # --------------------------------------------------

        features.append(mfcc)

        labels.append(emotion)

        actors.append(actor_name)

        file_names.append(
            str(file)
        )


        # Progress message
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

actors = np.array(actors)

file_names = np.array(file_names)


# --------------------------------------------------
# Save everything
# --------------------------------------------------

output_file = Path(
    "dataset/mfcc_features.pkl"
)


with open(
    output_file,
    "wb"
) as file:

    pickle.dump(
        {
            "features": features,
            "labels": labels,
            "actors": actors,
            "file_names": file_names
        },
        file
    )


# --------------------------------------------------
# Final information
# --------------------------------------------------

print("\nMFCC extraction completed!")

print(
    f"Total features: {len(features)}"
)

print(
    f"Feature shape: {features.shape}"
)

print(
    f"Labels: {len(labels)}"
)

print(
    f"Actors: {len(actors)}"
)

print(
    f"File names: {len(file_names)}"
)

print(
    f"Saved to: {output_file}"
)