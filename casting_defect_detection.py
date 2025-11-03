import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define paths to your dataset
dataset_path = './casting_512x512/'
image_size = (299, 299)  # For Xception model
batch_size = 32

# Create ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Create generators for training and validation datasets
train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, 'train'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(dataset_path, 'val'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Custom CNN Model
def create_custom_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(299, 299, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Xception Model (Transfer Learning)
def create_xception_model():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False  # Freeze base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and train custom CNN model
custom_cnn_model = create_custom_cnn()
history_cnn = custom_cnn_model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Save custom CNN model
custom_cnn_model.save('custom_cnn_model.h5')

# Create and train Xception transfer learning model
xception_model = create_xception_model()
history_xception = xception_model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Save Xception model
xception_model.save('xception_transfer_model.h5')

# Evaluate custom CNN model
cnn_val_loss, cnn_val_acc = custom_cnn_model.evaluate(val_generator)
print(f'Custom CNN Validation Loss: {cnn_val_loss}')
print(f'Custom CNN Validation Accuracy: {cnn_val_acc}')

# Evaluate Xception model
xception_val_loss, xception_val_acc = xception_model.evaluate(val_generator)
print(f'Xception Validation Loss: {xception_val_loss}')
print(f'Xception Validation Accuracy: {xception_val_acc}')

# Function to plot training history
def plot_history(history):
    if history is not None:
        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
    else:
        print("No history object available.")

# Plot the histories
plot_history(history_cnn)  # Custom CNN history
plot_history(history_xception)  # Xception history
