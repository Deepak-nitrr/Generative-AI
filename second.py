import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Check TensorFlow version and set memory growth
print(f"TensorFlow version: {tf.__version__}")

# Configure GPU memory growth if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define constants
IMG_WIDTH, IMG_HEIGHT = 256, 256
BATCH_SIZE = 1
NUM_SAMPLES = 100  # Number of synthetic samples to generate

# Generate synthetic dataset (instead of loading from files)
def generate_synthetic_dataset(num_samples):
    """Generate synthetic image pairs for training"""
    # Create input images (source)
    np.random.seed(42)  # For reproducibility
    input_images = np.random.rand(num_samples, IMG_HEIGHT, IMG_WIDTH, 3).astype(np.float32)
    
    # Create target images (simple transformation of input for demonstration)
    # In a real pix2pix application, these would be real paired images
    target_images = np.copy(input_images)
    
    # Apply some transformations to create target images
    for i in range(num_samples):
        # Add some noise and modify colors to create target
        target_images[i] = np.clip(input_images[i] * 0.8 + 0.2, 0, 1)
        # Add some pattern/structure
        if i % 2 == 0:
            target_images[i, :, :, 0] = np.clip(target_images[i, :, :, 0] + 0.1, 0, 1)
        
    return input_images, target_images

# Generate synthetic data
print("Generating synthetic dataset...")
train_images, train_labels = generate_synthetic_dataset(NUM_SAMPLES)
val_images, val_labels = generate_synthetic_dataset(20)  # Smaller validation set

print(f"Original data range - Train images: [{train_images.min():.3f}, {train_images.max():.3f}]")
print(f"Original data range - Train labels: [{train_labels.min():.3f}, {train_labels.max():.3f}]")

# Store original data for visualization
val_images_original = val_images.copy()
val_labels_original = val_labels.copy()

# Normalize images to [-1, 1] range
train_images = (train_images * 2.0) - 1.0
train_labels = (train_labels * 2.0) - 1.0
val_images = (val_images * 2.0) - 1.0
val_labels = (val_labels * 2.0) - 1.0

print(f"Normalized data range - Train images: [{train_images.min():.3f}, {train_images.max():.3f}]")
print(f"Normalized data range - Train labels: [{train_labels.min():.3f}, {train_labels.max():.3f}]")

# Define generator model
def define_generator():
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
    x = layers.Activation('relu')(x)
    outputs = layers.Conv2D(3, 5, padding='same')(x)
    outputs = layers.Activation('tanh')(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Define discriminator model
def define_discriminator():
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    labels = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = layers.Concatenate()([inputs, labels])
    x = layers.Conv2D(64, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(512, 4, strides=2, padding='same')(x)  # Added strides=2
    x = layers.LeakyReLU(0.2)(x)
    outputs = layers.Conv2D(1, 4, padding='same')(x)  # Changed kernel size to 4
    outputs = layers.Activation('sigmoid')(outputs)  # Added sigmoid activation
    model = keras.Model(inputs=[inputs, labels], outputs=outputs)
    return model

# Define pix2pix model
def define_pix2pix(generator, discriminator):
    discriminator.trainable = False
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    generated_images = generator(inputs)
    validity = discriminator([inputs, generated_images])
    model = keras.Model(inputs=inputs, outputs=[validity, generated_images])
    return model

# Build models
generator = define_generator()
discriminator = define_discriminator()
pix2pix = define_pix2pix(generator, discriminator)

# Print model summaries for debugging
print("Generator output shape:", generator.output_shape)
print("Discriminator output shape:", discriminator.output_shape)

# Compile models
discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5))
pix2pix.compile(loss=['binary_crossentropy', 'mae'], loss_weights=[1, 100], optimizer=keras.optimizers.Adam(0.0002, 0.5))

# Calculate discriminator output shape
disc_output_shape = discriminator.output_shape[1:]  # Remove batch dimension
print(f"Discriminator output shape (without batch): {disc_output_shape}")

# Train model
print("Starting training...")
for epoch in range(5):  # Reduced epochs for testing
    epoch_d_loss_real = []
    epoch_d_loss_fake = []
    epoch_g_loss = []
    
    for batch in range(min(10, len(train_labels) // BATCH_SIZE)):  # Limit batches for testing
        labels = train_labels[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]
        images = train_images[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]
        
        # Generate fake images
        generated_images = generator.predict(labels, verbose=0)
        
        # Create target arrays with correct shape
        real_targets = np.ones((BATCH_SIZE,) + disc_output_shape)
        fake_targets = np.zeros((BATCH_SIZE,) + disc_output_shape)
        
        # Train discriminator
        d_loss_real = discriminator.train_on_batch([labels, images], real_targets)
        d_loss_fake = discriminator.train_on_batch([labels, generated_images], fake_targets)
        
        # Train generator
        g_loss = pix2pix.train_on_batch(labels, [real_targets, images])
        
        epoch_d_loss_real.append(d_loss_real)
        epoch_d_loss_fake.append(d_loss_fake)
        epoch_g_loss.append(g_loss[0])
    
    print(f'Epoch {epoch+1}, D loss (real): {np.mean(epoch_d_loss_real):.4f}, D loss (fake): {np.mean(epoch_d_loss_fake):.4f}, G loss: {np.mean(epoch_g_loss):.4f}')

print("Training completed!")

# Evaluate model
print("Generating sample images...")
generated_images = generator.predict(val_labels[:5], verbose=0)  # Only generate 5 samples

# Create output directory
os.makedirs('generated_samples', exist_ok=True)

# Save some sample results
for i in range(min(5, len(generated_images))):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert from [-1, 1] to [0, 1] for display
    input_img = (val_labels[i] + 1) / 2
    generated_img = (generated_images[i] + 1) / 2
    target_img = val_images_original[i]  # Use original unormalized target
    
    # Clip values to valid range
    input_img = np.clip(input_img, 0, 1)
    generated_img = np.clip(generated_img, 0, 1)
    target_img = np.clip(target_img, 0, 1)
    
    axes[0].imshow(input_img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(generated_img)
    axes[1].set_title('Generated Image')
    axes[1].axis('off')
    
    axes[2].imshow(target_img)
    axes[2].set_title('Target Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'generated_samples/sample_{i+1}.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
print("Sample images saved to 'generated_samples/' directory")
print("Pix2Pix model training and evaluation completed successfully!")
