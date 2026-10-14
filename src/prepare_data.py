import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# --------------------------------------------------
# 1. Load extracted MFCC features
# --------------------------------------------------

with open("dataset/mfcc_features.pkl", "rb") as file:
    data = pickle.load(file)

features = data["features"]
labels = data["labels"]


# --------------------------------------------------
# 2. Display original data information
# --------------------------------------------------

print("Original dataset")
print("----------------")
print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")


# --------------------------------------------------
# 3. Convert emotion names to numbers
# --------------------------------------------------

label_encoder = LabelEncoder()

encoded_labels = label_encoder.fit_transform(labels)


print("\nEmotion labels:")
for number, emotion in enumerate(label_encoder.classes_):
    print(f"{number} - {emotion}")


# --------------------------------------------------
# 4. Split into training and testing data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    features,
    encoded_labels,
    test_size=0.2,
    random_state=42,
    stratify=encoded_labels
)


# --------------------------------------------------
# 5. Display split information
# --------------------------------------------------

print("\nData split completed!")
print(f"Training features: {X_train.shape}")
print(f"Testing features: {X_test.shape}")
print(f"Training labels: {y_train.shape}")
print(f"Testing labels: {y_test.shape}")