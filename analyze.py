import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from src.data_loader import generate_synthetic_data

TEST_SAMPLES = 50
TEST_SEED = 7  # different from the training seed, so these are unseen examples

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

if __name__ == '__main__':
    # 1. Load Model and Test Data
    if not os.path.exists('models/denoiser_model.keras'):
        print("Model not found. Please run train.py first.")
        sys.exit(1)

    model = tf.keras.models.load_model('models/denoiser_model.keras')
    X_noisy_test, X_clean_test = generate_synthetic_data(num_samples=TEST_SAMPLES, seed=TEST_SEED)

    # 2. Make Predictions
    denoised_imgs = model.predict(X_noisy_test, verbose=0)

    # 3. Analyze Performance
    input_snr = np.array([calculate_snr(c, n) for c, n in zip(X_clean_test, X_noisy_test)])
    output_snr = np.array([calculate_snr(c, d) for c, d in zip(X_clean_test, denoised_imgs)])

    print("-" * 36)
    print(f"{'Metric':<20} | {'Value':<10}")
    print("-" * 36)
    print(f"{'Test examples':<20} | {TEST_SAMPLES}")
    print(f"{'Avg input SNR':<20} | {input_snr.mean():.2f} dB")
    print(f"{'Avg output SNR':<20} | {output_snr.mean():.2f} dB")
    print(f"{'Avg improvement':<20} | {(output_snr - input_snr).mean():+.2f} dB")
    print("-" * 36)

    os.makedirs('results', exist_ok=True)
    metrics = {
        'test_examples': TEST_SAMPLES,
        'test_seed': TEST_SEED,
        'avg_input_snr_db': round(float(input_snr.mean()), 2),
        'avg_output_snr_db': round(float(output_snr.mean()), 2),
        'avg_improvement_db': round(float((output_snr - input_snr).mean()), 2),
        'min_improvement_db': round(float((output_snr - input_snr).min()), 2),
    }
    with open('results/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # 4. Visualize three examples: noisy input, clean target, model output
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for row in range(3):
        for col, (img, title) in enumerate([
            (X_noisy_test[row], f'Noisy input ({input_snr[row]:.1f} dB)'),
            (X_clean_test[row], 'Clean target'),
            (denoised_imgs[row], f'Denoised output ({output_snr[row]:.1f} dB)'),
        ]):
            axes[row, col].imshow(img.squeeze(), cmap='magma', vmin=0, vmax=1)
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis('off')
    fig.tight_layout()
    fig.savefig('results/result_comparison.png', dpi=110)
    print("Saved results/metrics.json and results/result_comparison.png")
