from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


# Load and Preprocess data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

model = tf.Sequential([
    layers.Conv2D(
        32, (3,3), 
        activation='relu', 
        padding ='same',
    input_shape=(32,32,3)
    ),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(62, (3,3), activation = 'relu' )
    ])


# Compile and train

model.compile(optimizer='adam',
        loss='sparse_categoriocal_crossentropy',
        metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

#Evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"Test Accuracy: {test_acc:.3f}")

#Plot accuracy
plt.plot(history['accuracy'], label = 'Train')
plt.plot(history.history['val_accuracy'], label= 'Validation')
plt.legend(); plt.title('Accuracy'); plt.show()

#model 
#def load_datasets(data_fir: str, image_size=(180, 180), batch_size: init = 32):
   # """
    
    #_summary_

   # Args:
    #    data_fir (str): _description_
     #   image_size (tuple, optional): _description_. Defaults to (180, 180).
      #  batch_size (init, optional): _description_. Defaults to 32.
    #"""