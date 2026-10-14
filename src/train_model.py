import pickle
import numpy as np

from pathlib import Path

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)
from tensorflow.keras.utils import to_categorical


# --------------------------------------------------
# Load MFCC features
# --------------------------------------------------

with open("dataset/mfcc_features.pkl", "rb") as file:
    data = pickle.load(file)

features = data["features"]
labels = data["labels"]


print("Dataset loaded")
print("----------------")
print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")


# --------------------------------------------------
# Find audio files
# --------------------------------------------------

dataset_path = Path("dataset/RAVDESS")

wav_files = sorted(dataset_path.rglob("*.wav"))


# --------------------------------------------------
# Create speaker-independent split
# Actors 01-20 = training
# Actors 21-24 = testing
# --------------------------------------------------

train_indices = []
test_indices = []

for index, file in enumerate(wav_files):

    actor_name = file.parent.name
    actor_number = int(actor_name.split("_")[1])

    if actor_number <= 20:
        train_indices.append(index)

    elif actor_number >= 21:
        test_indices.append(index)


print("\nSpeaker-independent split")
print("-------------------------")
print(f"Training files: {len(train_indices)}")
print(f"Testing files: {len(test_indices)}")


# --------------------------------------------------
# Encode emotion labels
# --------------------------------------------------

label_encoder = LabelEncoder()

encoded_labels = label_encoder.fit_transform(labels)

categorical_labels = to_categorical(encoded_labels)


print("\nEmotion classes:")

for number, emotion in enumerate(label_encoder.classes_):
    print(f"{number} - {emotion}")


# --------------------------------------------------
# Create training and testing data
# --------------------------------------------------

X_train = features[train_indices]
X_test = features[test_indices]

y_train = categorical_labels[train_indices]
y_test = categorical_labels[test_indices]


print("\nData split")
print("----------")
print(f"Training data: {X_train.shape}")
print(f"Testing data: {X_test.shape}")


# --------------------------------------------------
# Build CNN
# --------------------------------------------------

model = Sequential([
    Input(shape=(40, 130, 1)),

    Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    Dropout(0.25),

    Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    Dropout(0.25),

    Flatten(),

    Dense(
        64,
        activation="relu"
    ),

    BatchNormalization(),

    Dropout(0.5),

    Dense(
        8,
        activation="softmax"
    )
])


# --------------------------------------------------
# Compile model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nSpeaker-independent CNN model:")
model.summary()


# --------------------------------------------------
# Callbacks
# --------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001
)


# --------------------------------------------------
# Train CNN
# --------------------------------------------------

print("\nStarting speaker-independent CNN training...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=32,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)


# --------------------------------------------------
# Evaluate model
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print("\nTraining completed!")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# --------------------------------------------------
# Save model
# --------------------------------------------------

model.save(
    "models/emotion_cnn_speaker_independent.keras"
)

print("\nSpeaker-independent CNN model saved successfully!")
print(
    "Location: models/emotion_cnn_speaker_independent.keras"
)


# --------------------------------------------------
# Save label encoder
# --------------------------------------------------

with open(
    "models/label_encoder_speaker_independent.pkl",
    "wb"
) as file:

    pickle.dump(
        label_encoder,
        file
    )

print("Label encoder saved successfully!")