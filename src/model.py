import tensorflow as tf
from tensorflow.keras import layers, models

def build_denoising_autoencoder(input_shape):
    """
    Constructs a Convolutional Autoencoder.
    Input: Noisy Spectrogram -> Output: Clean Spectrogram
    """
    model = models.Sequential()

    # --- Encoder (Compressing the signal) ---
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))

    # --- Decoder (Reconstructing the signal) ---
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))

    # Output layer (1 channel for grayscale spectrogram)
    model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
