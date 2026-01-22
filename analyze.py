import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from src.data_loader import generate_synthetic_data

def calculate_snr(clean, noisy):
    """
    Compute Signal-to-Noise Ratio (SNR).
    Formula: 10 * log10(Signal_Power / Noise_Power)
    """
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum((clean - noisy) ** 2)

    # Avoid division by zero
    if noise_power == 0:
        return 100 

    return 10 * np.log10(signal_power / noise_power)

# 1. Load Model and Test Data
if not os.path.exists('models/denoiser_model.h5'):
    print("Model not found. Please run train.py first.")
    exit()

model = tf.keras.models.load_model('models/denoiser_model.h5')
X_noisy_test, X_clean_test = generate_synthetic_data(num_samples=10)

# 2. Make Predictions
denoised_imgs = model.predict(X_noisy_test)

# 3. Analyze Performance
print("-" * 30)
print(f"{'Metric':<15} | {'Value':<10}")
print("-" * 30)

avg_input_snr = 0
avg_output_snr = 0

for i in range(len(X_noisy_test)):
    original_snr = calculate_snr(X_clean_test[i], X_noisy_test[i])
    improved_snr = calculate_snr(X_clean_test[i], denoised_imgs[i])

    avg_input_snr += original_snr
    avg_output_snr += improved_snr

print(f"{'Avg Input SNR':<15} | {avg_input_snr/10:.2f} dB")
print(f"{'Avg Output SNR':<15} | {avg_output_snr/10:.2f} dB")
print("-" * 30)

# 4. Visualize and Save Result
if not os.path.exists('output'):
    os.makedirs('output')

idx = 0
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(X_noisy_test[idx].squeeze(), cmap='gray')
axes[0].set_title('Noisy Input')
axes[1].imshow(X_clean_test[idx].squeeze(), cmap='gray')
axes[1].set_title('Original Clean')
axes[2].imshow(denoised_imgs[idx].squeeze(), cmap='gray')
axes[2].set_title('Denoised Output (AI)')
plt.savefig('output/result_comparison.png')
print("Visual comparison saved to output/result_comparison.png")
