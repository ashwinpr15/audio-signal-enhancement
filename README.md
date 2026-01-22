# Audio Denoising Autoencoder

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

A Convolutional Denoising Autoencoder (DAE) implemented in TensorFlow/Keras. This model removes additive Gaussian noise from unstructured audio data by learning robust feature representations in the time-frequency domain (spectrograms).

## ⚡ Technical Overview

This project tackles signal clarity enhancement using a supervised deep learning approach. Instead of traditional DSP filters, it uses a **Convolutional Encoder-Decoder** architecture to map noisy inputs to clean targets.

* **Input:** 128x128 Log-Mel Spectrograms (simulated) + Gaussian Noise.
* **Model:** 4-layer CNN Autoencoder with downsampling (Encoder) and upsampling (Decoder).
* **Loss Function:** Mean Squared Error (MSE) for pixel-wise reconstruction.
* **Metric:** Signal-to-Noise Ratio (SNR) for signal clarity benchmarking.

## 📊 Performance Benchmarks

Statistical analysis was performed on the test set to quantify signal recovery.

| Metric | Input (Noisy) | Output (Denoised) | Improvement |
| :--- | :--- | :--- | :--- |
| **Average SNR** | ~2.5 dB | ~14.2 dB | **+11.7 dB** |
| **MSE Loss** | 0.045 | 0.008 | **-82%** |

*> **Note:** Higher SNR (Signal-to-Noise Ratio) indicates better signal clarity and reduced background interference.*

## 📂 Repository Structure

```text
.
├── src/
│   ├── model.py         # CNN Autoencoder architecture (Keras)
│   ├── data_loader.py   # Synthetic spectrogram generator (NumPy)
├── train.py             # Training pipeline and model checkpointing
├── analyze.py           # Statistical evaluation (SNR calculation)
├── requirements.txt     # Dependencies
└── README.md            # Documentation





