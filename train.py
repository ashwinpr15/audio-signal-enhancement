import os
from src.model import build_denoising_autoencoder
from src.data_loader import generate_synthetic_data
import matplotlib.pyplot as plt

# 1. Settings
IMG_SHAPE = (128, 128, 1)
EPOCHS = 10
BATCH_SIZE = 16

# 2. Load Data
print("Loading/Generating Dataset...")
X_noisy, X_clean = generate_synthetic_data(num_samples=500, img_shape=(128, 128))
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
if not os.path.exists('models'):
    os.makedirs('models')
autoencoder.save('models/denoiser_model.h5')
print("Model saved to models/denoiser_model.h5")

# 6. Save Training History Plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Model Convergence')
plt.savefig('output/training_curve.png')
print("Training curve saved to output/training_curve.png")
