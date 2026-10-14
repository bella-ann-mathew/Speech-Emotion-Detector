import pickle
import numpy as np

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
# Load original MFCC features
# --------------------------------------------------

with open(
    "dataset/mfcc_features.pkl",
    "rb"
) as file:

    original_data = pickle.load(file)


original_features = original_data["features"]
original_labels = original_data["labels"]
original_actors = original_data["actors"]


# --------------------------------------------------
# Load augmented MFCC features
# --------------------------------------------------

with open(
    "dataset/mfcc_augmented_features.pkl",
    "rb"
) as file:

    augmented_data = pickle.load(file)


augmented_features = augmented_data["features"]
augmented_labels = augmented_data["labels"]


print("Features loaded")
print("----------------")
print(
    f"Original features: {original_features.shape}"
)
print(
    f"Augmented features: {augmented_features.shape}"
)


# --------------------------------------------------
# Create speaker-independent split
# --------------------------------------------------

train_mask = np.array([
    actor_number <= 20
    for actor_number in [
        int(actor.split("_")[1])
        for actor in original_actors
    ]
])

test_mask = np.array([
    actor_number >= 21
    for actor_number in [
        int(actor.split("_")[1])
        for actor in original_actors
    ]
])


# --------------------------------------------------
# Select ONLY training actors from original data
# --------------------------------------------------

original_train_features = original_features[
    train_mask
]

original_train_labels = original_labels[
    train_mask
]


# --------------------------------------------------
# Select ONLY testing actors from original data
# --------------------------------------------------

test_features = original_features[
    test_mask
]

test_labels = original_labels[
    test_mask
]


print("\nSpeaker-independent split")
print("-------------------------")
print(
    f"Original training features: "
    f"{original_train_features.shape}"
)
print(
    f"Augmented training features: "
    f"{augmented_features.shape}"
)
print(
    f"Testing features: "
    f"{test_features.shape}"
)


# --------------------------------------------------
# Combine original + augmented training data
# --------------------------------------------------

X_train = np.concatenate(
    [
        original_train_features,
        augmented_features
    ],
    axis=0
)

combined_train_labels = np.concatenate(
    [
        original_train_labels,
        augmented_labels
    ],
    axis=0
)


print("\nCombined training dataset")
print("-------------------------")
print(
    f"Training features: {X_train.shape}"
)
print(
    f"Training labels: "
    f"{combined_train_labels.shape}"
)


# --------------------------------------------------
# Encode emotion labels
# --------------------------------------------------

label_encoder = LabelEncoder()

encoded_train_labels = label_encoder.fit_transform(
    combined_train_labels
)

encoded_test_labels = label_encoder.transform(
    test_labels
)


y_train = to_categorical(
    encoded_train_labels
)

y_test = to_categorical(
    encoded_test_labels
)


print("\nEmotion classes")

for number, emotion in enumerate(
    label_encoder.classes_
):

    print(
        f"{number} - {emotion}"
    )


# --------------------------------------------------
# Display final dataset sizes
# --------------------------------------------------

print("\nFinal dataset")
print("-------------")
print(
    f"Training data: {X_train.shape}"
)
print(
    f"Testing data: {test_features.shape}"
)


# --------------------------------------------------
# Create CNN model
# --------------------------------------------------

model = Sequential([

    Input(
        shape=(40, 130, 1)
    ),

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


print("\nSpeaker-independent augmented CNN model:")
model.summary()


# --------------------------------------------------
# Training callbacks
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
# Train model
# --------------------------------------------------

print(
    "\nStarting corrected augmented CNN training...\n"
)


history = model.fit(

    X_train,

    y_train,

    validation_data=(
        test_features,
        y_test
    ),

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
    test_features,
    y_test,
    verbose=0
)


print("\nTraining completed!")

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)


# --------------------------------------------------
# Save model
# --------------------------------------------------

model.save(
    "models/emotion_cnn_augmented_corrected.keras"
)


print(
    "\nCorrected augmented CNN model saved successfully!"
)

print(
    "Location: "
    "models/emotion_cnn_augmented_corrected.keras"
)


# --------------------------------------------------
# Save label encoder
# --------------------------------------------------

with open(
    "models/label_encoder_augmented_corrected.pkl",
    "wb"
) as file:

    pickle.dump(
        label_encoder,
        file
    )


print(
    "Label encoder saved successfully!"
)