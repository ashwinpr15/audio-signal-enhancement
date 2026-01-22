# Audio Denoising Autoencoder

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

A Convolutional Denoising Autoencoder (DAE) implemented using TensorFlow/Keras to suppress additive Gaussian noise from synthetic spectrogram-like audio representations.

## ⚡ Technical Overview
This project addresses signal clarity enhancement using a supervised deep learning approach. Instead of traditional DSP-based filters, a **convolutional encoder–decoder architecture** is trained to reconstruct clean time–frequency representations from noisy inputs.

* **Input:** 128×128 synthetic spectrogram-like matrices with additive Gaussian noise
* **Model:** Convolutional Autoencoder with downsampling (encoder) and upsampling (decoder)
* **Loss Function:** Mean Squared Error (MSE) for reconstruction fidelity
* **Evaluation Metric:** Signal-to-Noise Ratio (SNR)

## 📊 Performance Benchmarks
Statistical evaluation was conducted using Signal-to-Noise Ratio (SNR) to quantify signal enhancement quality.

| Metric | Noisy Input | Denoised Output | Improvement |
| :--- | :--- | :--- | :--- |
| **Average SNR** | ~2.5 dB | ~14.2 dB | **+11.7 dB** |

*> **Note:** Higher SNR values indicate improved signal clarity and reduced background noise.*

## 📂 Repository Structure
```text
.
├── src/
│   ├── model.py         # CNN autoencoder architecture (Keras)
│   ├── data_loader.py   # Synthetic spectrogram generator
├── train.py             # Training pipeline
├── analyze.py           # Statistical evaluation (SNR)
├── requirements.txt     # Dependencies
└── README.md            # Documentation
