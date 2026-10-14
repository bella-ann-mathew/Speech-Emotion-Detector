
import pickle
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("models/emotion_cnn_augmented_corrected.keras")

# Load the matching label encoder
with open("models/label_encoder_augmented_corrected.pkl", "rb") as file:
    label_encoder = pickle.load(file)

# Enter the path to your WAV audio file
audio_path = input("Enter the path of your WAV audio file: ").strip().strip('"')

# Extract MFCC features from the audio
audio, sample_rate = librosa.load(audio_path, sr=16000)

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=40
)

# Make the feature length 130 frames
if mfcc.shape[1] < 130:
    mfcc = np.pad(
        mfcc,
        ((0, 0), (0, 130 - mfcc.shape[1])),
        mode="constant"
    )
else:
    mfcc = mfcc[:, :130]

# Prepare input for the CNN
mfcc = mfcc[np.newaxis, ..., np.newaxis]

# Predict the emotion
predictions = model.predict(mfcc, verbose=0)
predicted_index = np.argmax(predictions[0])
predicted_emotion = label_encoder.inverse_transform([predicted_index])[0]
confidence = float(predictions[0][predicted_index]) * 100

print("\nSpeech Emotion Recognition Result")
print("--------------------------------")
print("Predicted emotion:", predicted_emotion)
print(f"Model confidence: {confidence:.2f}%")
