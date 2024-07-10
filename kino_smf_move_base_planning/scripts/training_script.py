import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(28, 28), color_mode="grayscale")
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array


# Load your dataset (paths to images and labels)
train_images_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]
train_labels = [[1, 0, 0, ...], [0, 1, 0, ...], ...]  # One-hot encoded labels

# Preprocess the images and labels
train_images = np.array(
    [load_and_preprocess_image(path) for path in train_images_paths]
)
train_labels = np.array(train_labels)

# Define the CNN model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, (5, 5), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (5, 5), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(10),  # Assuming 10 classes
    ]
)

# Compile the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Create a checkpoint callback to save the model
checkpoint_path = "training_checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1, period=1
)  # Save every epoch

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, callbacks=[cp_callback])

# Save the final model
model.save("saved_model/my_model")
