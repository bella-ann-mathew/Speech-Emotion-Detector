
from pathlib import Path
import pickle

import librosa
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Flask web application
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates")
)

# Load the trained model and label encoder
model = load_model(MODEL_DIR / "emotion_cnn_augmented_corrected.keras")

with open(MODEL_DIR / "label_encoder_augmented_corrected.pkl", "rb") as file:
    label_encoder = pickle.load(file)


def predict_emotion(audio_path):
    audio, sample_rate = librosa.load(audio_path, sr=16000)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    if mfcc.shape[1] < 130:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, 130 - mfcc.shape[1])),
            mode="constant"
        )
    else:
        mfcc = mfcc[:, :130]

    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    predictions = model.predict(mfcc, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))

    emotion = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(predictions[0][predicted_index]) * 100

    return emotion, confidence


@app.route("/", methods=["GET", "POST"])
def home():
    emotion = None
    confidence = None
    error = None

    if request.method == "POST":
        audio_file = request.files.get("audio")

        if not audio_file or not audio_file.filename:
            error = "Please select a WAV audio file."
        elif not audio_file.filename.lower().endswith(".wav"):
            error = "Please upload a WAV file."
        else:
            try:
                import tempfile

                with tempfile.TemporaryDirectory() as temp_dir:
                    audio_path = Path(temp_dir) / "uploaded_audio.wav"
                    audio_file.save(audio_path)
                    emotion, confidence = predict_emotion(audio_path)

            except Exception as exc:
                print("Prediction error:", exc)
                error = "Could not process this audio file. Please try another WAV recording."

    return render_template(
        "index.html",
        emotion=emotion,
        confidence=confidence,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
