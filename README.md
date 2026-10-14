# Speech Emotion Recognition

## Project Overview
This project detects human emotions from speech audio using Machine Learning and Deep Learning. It extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio recordings and uses a Convolutional Neural Network (CNN) to classify the emotion.

## Objectives
- Recognize emotions from speech recordings.
- Extract meaningful audio features using MFCC.
- Train and evaluate a CNN model.
- Provide a simple web interface for uploading audio and viewing predictions.

## Technologies Used
- Python
- TensorFlow and Keras
- Librosa
- NumPy
- Scikit-learn
- Matplotlib
- Flask
- HTML, CSS, and JavaScript

## Dataset
The project uses the RAVDESS speech dataset.

The model recognizes eight emotion classes:
- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## Methodology
1. Load speech audio recordings.
2. Extract 40 MFCC coefficients from each recording.
3. Pad or crop the extracted features to a fixed length of 130 frames.
4. Augment training audio with background noise.
5. Train a CNN model using the training speakers.
6. Evaluate the model using recordings from different speakers.
7. Use the trained model to predict emotions through a Flask web application.

## Model Evaluation
The speaker-independent evaluation achieved **46.67% accuracy** on 240 test recordings from actors 21–24.

This evaluation uses speakers excluded from training, providing a more realistic measure of how the model performs on unseen speakers. The results also indicate that further improvements are needed.

## How to Run the Project

### 1. Activate the virtual environment
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

### 3. Start the web application
```powershell
python src/app.py
```

### 4. Open the application
Open this address in your browser:

http://127.0.0.1:5000

Upload a WAV audio file and click **Analyze Emotion** to view the predicted emotion and confidence score.

## Project Structure
- `src/` — Feature extraction, model training, evaluation, prediction, and Flask application scripts.
- `dataset/` — Audio recordings and extracted MFCC features.
- `models/` — Trained model, label encoder, and evaluation outputs.
- `templates/` — HTML template for the web interface.
- `requirements.txt` — Python dependency versions.
- `README.md` — Project documentation.

## Future Improvements
- Improve accuracy using a larger and more diverse dataset.
- Experiment with different CNN, RNN, or LSTM architectures.
- Improve generalization to unseen speakers.
- Add support for more audio formats and real-time microphone input.

## Conclusion
This project demonstrates a speech emotion recognition pipeline using MFCC feature extraction, CNN-based classification, and a Flask web interface. It provides a foundation for further experiments in speech emotion recognition.
