import numpy as np

def generate_synthetic_data(num_samples=100, img_shape=(128, 128)):
    """
    Generates synthetic 'spectrogram-like' data for demonstration.
    Returns: X_noisy (Input), X_clean (Target)
    """
    X_clean = []
    X_noisy = []

    for _ in range(num_samples):
        # 1. Create a clean pattern (simulating a signal)
        clean_img = np.zeros(img_shape)
        row_idx = np.random.randint(0, img_shape[0], 5)
        clean_img[row_idx, :] = 1.0  # Horizontal lines representing signal

        # 2. Add Gaussian Noise
        noise = np.random.normal(loc=0.0, scale=0.3, size=img_shape)
        noisy_img = clean_img + noise

        # 3. Clip values to be between 0 and 1
        noisy_img = np.clip(noisy_img, 0., 1.)

        X_clean.append(clean_img)
        X_noisy.append(noisy_img)

    # Reshape for TensorFlow (samples, height, width, channels)
    X_clean = np.array(X_clean)[..., np.newaxis]
    X_noisy = np.array(X_noisy)[..., np.newaxis]

    return X_noisy, X_clean
