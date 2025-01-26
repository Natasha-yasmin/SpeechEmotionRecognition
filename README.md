
# Speech Emotion Recognition using RAVDESS Dataset

A comprehensive pipeline for speech emotion recognition using the RAVDESS Emotional Speech Audio dataset, featuring data processing, model training with fastai, and live prediction capabilities.

## Table of Contents
1. [Dataset Introduction](#dataset-introduction)
2. [Google Drive Setup](#google-drive-setup)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Extraction](#feature-extraction)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Live Audio Prediction](#live-audio-prediction)

---

## Dataset Introduction <a name="dataset-introduction"></a>
**RAVDESS Emotional Speech Audio Dataset** from Kaggle:
- 1440 audio files (24 professional actors)
- 7 emotional categories + neutral
- Two intensity levels (normal/strong)
- Balanced gender representation

```python
{
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# Download the RAVDESS Emotional Speech Audio Dataset",
        "Files: 1440 audio samples (60 trials/actor × 24 actors)",
        "Emotions: calm, happy, sad, angry, fearful, surprise, disgust + neutral"
      ]
    }
  ]
}
```

---

## Google Drive Setup <a name="google-drive-setup"></a>
```python
from google.colab import drive
drive.mount('/content/drive')

# Unzip dataset
#!unzip "/content/drive/MyDrive/.../RAVDESS.zip" -d "/content/.../Dataset/"
```

---

## Data Preprocessing <a name="data-preprocessing"></a>
### Label Extraction Class
```python
class FetchLabel():
    def get_emotion(self, file_path):
        # Extract emotion from filename pattern
        if item[6:-16] == '02' and int(item[18:-4])%2 == 0:
            return 'female_calm'
        # ... (full conditional logic for all emotions)
```

### Dependency Installation
```python
!pip install sounddevice
!sudo apt-get install libportaudio2
```

---

## Feature Extraction <a name="feature-extraction"></a>
### Audio Visualization
```python
# Waveform visualization
data, sampling_rate = librosa.load('audio_file.wav')
plt.figure(figsize=(40, 5))
librosa.display.waveplot(data, sr=sampling_rate)

# Mel-spectrogram conversion
mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=100)
mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
```

### Data Pipeline
```python
# Create ImageDataLoaders
dls = ImageDataLoaders.from_folder(train_path, 
                                  valid_pct=0.2, 
                                  seed=42,
                                  num_workers=0)
```

---

## Model Training <a name="model-training"></a>
### ResNet34 Architecture
```python
learn = cnn_learner(dls, 
                   models.resnet34, 
                   loss_func=CrossEntropyLossFlat(),
                   metrics=accuracy)
```

### Learning Rate Finder
```python
lr = learn.lr_find(suggest_funcs=(minimum, steep))
print(f"Optimal LR: {lr.steep:.2e}")
```

### Training Process
```python
learn.fit(20, lr=5.25e-03)
```

| Epoch | Train Loss | Valid Loss | Accuracy | Time  |
|-------|------------|------------|----------|-------|
| 0     | 0.202879   | 2.481177   | 0.507937 | 00:15 |
| ...   | ...        | ...        | ...      | ...   |
| 19    | 0.194418   | 2.478409   | 0.619048 | 00:15 |

---

## Model Evaluation <a name="model-evaluation"></a>
### Confusion Matrix
```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
```

![Confusion Matrix](data:image/png;base64,...)

### Model Export
```python
learn.export('speech_02.pkl')
```

---

## Live Audio Prediction <a name="live-audio-prediction"></a>
### Prediction Pipeline
```python
# Load pretrained model
model = load_learner('/path/to/speech_02.pkl')

# Process live audio
y, sr = librosa.load('live_audio.wav')
mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
# ... (preprocessing steps)

# Make prediction
live_pred = plt.imread('processed_image.jpg')
emotion = model.predict(live_pred)[0]
print(f"Predicted Emotion: {emotion}")
```

### Example Predictions
1. Recording 1: calm (87% confidence)
2. Recording 3: disgust (63% confidence)
3. Recording 5: neutral (72% confidence)

---

## Conclusion
Final model achieves 61.9% validation accuracy with ResNet34 architecture. The pipeline demonstrates complete workflow from raw audio processing to live emotion prediction. Future improvements could include data augmentation and more sophisticated architectures like Transformer networks.
