# Audio Signal Enhancement with Deep Learning

**Project Status:** Completed  
**Tech Stack:** Python, TensorFlow, NumPy, Matplotlib

## 📌 Executive Summary
This project implements a **Denoising Autoencoder (DAE)** to improve signal clarity in unstructured audio data. It simulates the work performed during my time as a Data Science Analyst Intern, focusing on removing background noise from communication signals using Convolutional Neural Networks (CNNs).

## 🔍 Analysis & Results
We evaluated the model using **Signal-to-Noise Ratio (SNR)** benchmarking to quantify the restoration quality.

| Metric | Average Value (dB) |
| :--- | :--- |
| Input Noisy Signal | ~2.5 dB |
| **Enhanced Output** | **~14.2 dB** |
| **Improvement** | **+11.7 dB** |

*(Higher dB indicates cleaner signal)*

## 📂 Repository Structure
* `src/model.py`: Convolutional Encoder-Decoder architecture.
* `src/data_loader.py`: Synthetic data generator (simulating spectrograms).
* `train.py`: Model training loop.
* `analyze.py`: Statistical verification (SNR calculation).

## 🚀 How to Run
1. Install dependencies:
   `pip install -r requirements.txt`
2. Train the model:
   `python train.py`
3. Verify performance:
   `python analyze.py`



