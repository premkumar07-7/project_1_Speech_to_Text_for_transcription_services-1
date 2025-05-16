# project_1_Speech_to_Text_for_transcription_services-1

---

## 📦 **1. Installation of Dependencies**

```python
!pip install kaggle
!pip install transformers torchaudio librosa noisereduce
!pip install openai-whisper
!pip install datasets
```

---

## 📤 **2. Upload Dataset**

```python
from google.colab import files
files.upload()
```

---

## 🎧 **3. Data Cleaning (Audio Preprocessing)**

* **Steps:**

  1. Load audio with `librosa`
  2. Trim silence
  3. Normalize signal
  4. Display cleaned waveform

```python
audio_path = '/sp01_street_sn5.wav'
y, sr = librosa.load(audio_path, sr=None)
y_clean, _ = librosa.effects.trim(y)
y_normalized = librosa.util.normalize(y_clean)

# Plot waveform
librosa.display.waveshow(y_normalized, sr=sr)
```

---

## 📊 **4. Data Analysis (Spectrogram)**

```python
D = librosa.amplitude_to_db(np.abs(librosa.stft(y_normalized)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
```

---

## 🧪 **5. Evaluation: Word Error Rate (WER)**

* Using `jiwer` to compare actual vs predicted transcriptions

```python
!pip install jiwer
from jiwer import wer

actual_transcription = "your actual transcription here"
predicted_transcription = "your predicted transcription here"
wer_score = wer(actual_transcription, predicted_transcription)
```

---

## 📈 **6. Visualization of WER (Bar Chart)**

```python
actual_transcriptions = ["hello how are you", "i need help with my task", "please call me back"]
predicted_transcriptions = ["hello how are you", "i need help with my cost", "please coll me back"]
wer_scores = [wer(act, pred) for act, pred in zip(actual_transcriptions, predicted_transcriptions)]
```

---

## 🔤 **7. Homophone Analysis**

* Counts occurrences of commonly misrecognized words.

```python
homophones_dict = {
    "there": ["their", "they're", "there"],
    "hear": ["here"],
    "to": ["too", "two"],
    "right": ["write"],
    "sea": ["see"],
    "bare": ["bear"]
}

def count_homophones(text, homophones_dict):
    ...
```

* Plotted using a **bar chart** showing homophone usage frequency across predicted transcriptions.

