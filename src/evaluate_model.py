
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import load_model


# --------------------------------------------------
# Load original MFCC features and actor information
# --------------------------------------------------

with open("dataset/mfcc_features.pkl", "rb") as file:
    data = pickle.load(file)

features = data["features"]
labels = data["labels"]
actors = data["actors"]

print("Dataset loaded successfully!")
print("-----------------------------")
print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")


# --------------------------------------------------
# Select only unseen test speakers: Actors 21-24
# --------------------------------------------------

actor_numbers = np.array([
    int(actor.split("_")[1])
    for actor in actors
])

test_mask = actor_numbers >= 21

X_test = features[test_mask]
y_test = labels[test_mask]

print("\nSpeaker-independent test dataset")
print("--------------------------------")
print(f"Testing recordings: {len(X_test)}")
print(f"Testing features: {X_test.shape}")


# --------------------------------------------------
# Load the saved label encoder
# --------------------------------------------------

with open(
    "models/label_encoder_augmented_corrected.pkl",
    "rb"
) as file:
    label_encoder = pickle.load(file)


# --------------------------------------------------
# Encode actual emotion labels
# --------------------------------------------------

actual_classes = label_encoder.transform(y_test)


# --------------------------------------------------
# Load the corrected CNN model
# --------------------------------------------------

model = load_model(
    "models/emotion_cnn_augmented_corrected.keras"
)

print("\nCorrected CNN model loaded successfully!")


# --------------------------------------------------
# Predict emotions
# --------------------------------------------------

predictions = model.predict(
    X_test,
    verbose=0
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)


# --------------------------------------------------
# Classification report
# --------------------------------------------------

emotion_names = label_encoder.classes_

print("\nClassification Report")
print("---------------------")

report = classification_report(
    actual_classes,
    predicted_classes,
    labels=np.arange(len(emotion_names)),
    target_names=emotion_names,
    zero_division=0
)

print(report)

with open("models/classification_report.txt", "w") as file:
    file.write(report)

print("Classification report saved to models/classification_report.txt")



# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    actual_classes,
    predicted_classes,
    labels=np.arange(len(emotion_names))
)

print("\nConfusion Matrix")
print("----------------")
print(cm)


# --------------------------------------------------
# Display confusion matrix
# --------------------------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=emotion_names
)

display.plot(
    xticks_rotation=45
)


plt.title("Speaker-Independent Emotion Recognition")
plt.tight_layout()

# Save the confusion matrix image
plt.savefig("models/confusion_matrix.png", dpi=300, bbox_inches="tight")
print("\nConfusion matrix saved to models/confusion_matrix.png")

plt.show()

