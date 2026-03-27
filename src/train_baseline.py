from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

# Load and Preprocess data
train_dir = "data/processed/train"
val_dir = "data/processed/val"
test_dir = "data/processed/test"

#Label the images 
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(180,180),
    batch_size= 64
     )

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(180,180),
    batch_size=64
    )

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(180,180),
    batch_size=64
)

# Class Info
class_names = train_ds.class_names
num_classes = len(class_names)

print ("Classes:", class_names)
print("Number of Classes:", num_classes)

# Normalise images
normalise = layers.Rescaling(1.0 /255)

train_ds = train_ds.map(lambda images, labels: (normalise(images), labels))
val_ds = val_ds.map(lambda images, labels: (normalise(images), labels))
test_ds = test_ds.map(lambda images, labels: (normalise(images), labels))

# Build Baseline CNN 

model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(180, 180, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

# Compile and train

model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

# Train Model
history = model.fit(train_ds, epochs=10,
                    validation_data=(val_ds))

# Evaluate
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test Accuracy: {test_acc:.3f}")

with open(f"results_{timestamp}.txt", "w") as f:
    f.write(f"Train Accuracy: {history.history['accuracy'][-1]:.4f}\n")
    f.write(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")

# Plot accuracy
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.legend()
plt.title("Accuracy")
plt.savefig(f"results/accuracy_{timestamp}.png")
plt.show()

#model 
#def load_datasets(data_fir: str, image_size=(180, 180), batch_size: init = 32):
   # """
    
    #_summary_

   # Args:
    #    data_fir (str): _description_
     #   image_size (tuple, optional): _description_. Defaults to (180, 180).
      #  batch_size (init, optional): _description_. Defaults to 32.
    #"""