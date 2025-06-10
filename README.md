# 🔥 FireRedTrainAlign: Chinese Speech Recognition Training and Inference Toolkit

This project provides training and inference code for the [FireRedTeam/FireRedASR-AED-L](https://huggingface.co/FireRedTeam/FireRedASR-AED-L) model, supporting high-quality Chinese speech transcription tasks.
The model adopts an **Attention-based Encoder-Decoder (AED)** architecture, suitable for Mandarin and certain dialects.

---

## 🚀 Project Features

* ✅ Supports the `FireRedASR-AED-L` model on Hugging Face
* ✅ Provides a complete training pipeline (data preprocessing, model training, and application)
* ✅ Supports beam search decoding and batch inference for multiple audio files
* ✅ Optimized for medium to long speech segments, applicable in various Chinese speech scenarios

---

## ⚙️ Installation & Environment Requirements

### 1️⃣ Create a Virtual Environment (Conda recommended)

```bash
git clone https://github.com/AUSTIN2526/FireRedTrainAlign.git
cd FireRedTrainAlign

conda create -n firered_asr python=3.10
conda activate firered_asr

pip install -r requirements.txt
```

### 2️⃣ Additional Dependencies

* `ffmpeg`: Required for audio format conversion and resampling
* `CUDA` (optional): For GPU-accelerated training or inference

---

## 🎯 Notes & Limitations

* It is recommended that audio length **does not exceed 60 seconds** to avoid decoding errors or repeated phrases
* Audio longer than **200 seconds** may cause errors in positional encoding
* Batch inference and training are **not fully tested** yet
* Please place the `FireRedASR-AED-L` model in the project directory for use

---

## 📁 Preparing Training Data

Please prepare a `.tsv or .txt` text file in the following format, where each line represents one audio sample and its corresponding transcription:

```
ID\tTranscription text
ID\tTranscription text
ID\tTranscription text
...
```

---

## 📊 Model Performance (from the official paper)

| Model Name       | Parameters | AISHELL1 | AISHELL2 | WS Net | WS Meeting | Avg. CER  |
| ---------------- | ---------- | -------- | -------- | ------ | ---------- | --------- |
| FireRedASR-AED-L | 1.1B       | 0.55%    | 2.52%    | 4.88%  | 4.76%      | **3.18%** |

## 🔗 **Speech-to-Text Integration: Supports `med-voice-SHI-detector`**

This project can be integrated with [med-voice-SHI-detector](https://github.com/AUSTIN2526/med-voice-SHI-detector) to automatically transcribe medical voice data into text and perform SHI (Speech-based Health Information) de-identification. The overall processing workflow is as follows:

1. **Speech-to-Text (ASR):**
   Use `med-voice-SHI-detector` to convert Chinese medical audio files into text.

2. **SHI De-identification:**
   Input the transcribed text into this project to identify and mask sensitive information such as names, healthcare institutions, and ID numbers.

3. **Output Format:**
   Generates de-identified text records suitable for downstream NLP tasks or data analysis.

👉 **Installation & Usage**
Please refer to the [official documentation of `med-voice-SHI-detector`](https://github.com/AUSTIN2526/med-voice-SHI-detector) for detailed instructions.


