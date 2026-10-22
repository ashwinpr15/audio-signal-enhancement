import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from src.model import build_denoising_autoencoder
from src.data_loader import generate_synthetic_data

# 1. Settings
IMG_SHAPE = (128, 128, 1)
EPOCHS = 10
BATCH_SIZE = 16
SEED = 42
tf.keras.utils.set_random_seed(SEED)

# 2. Load Data
print("Loading/Generating Dataset...")
X_noisy, X_clean = generate_synthetic_data(num_samples=500, img_shape=(128, 128), seed=SEED)
print(f"Data Shape: {X_noisy.shape}")

# 3. Build Model
print("Building Model...")
autoencoder = build_denoising_autoencoder(IMG_SHAPE)
autoencoder.summary()

# 4. Train
print("Starting Training...")
history = autoencoder.fit(
    X_noisy, X_clean,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.1
)

# 5. Save Model
os.makedirs('models', exist_ok=True)
autoencoder.save('models/denoiser_model.keras')
print("Model saved to models/denoiser_model.keras")

# 6. Save Training History Plot
os.makedirs('results', exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, EPOCHS + 1), history.history['loss'], marker='o', label='Train loss')
ax.plot(range(1, EPOCHS + 1), history.history['val_loss'], marker='o', label='Validation loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Binary cross-entropy')
ax.set_title('Training curve')
ax.legend()
fig.tight_layout()
fig.savefig('results/training_curve.png', dpi=120)
print("Training curve saved to results/training_curve.png")
