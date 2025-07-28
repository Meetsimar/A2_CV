import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# Set the directory paths for training and testing data
train_dir = 'train'
test_dir = 'test'

# File path to save the best model during training
model_path = 'best_cat_dog_model.h5'

# Image size to resize all images for consistency
img_size = (150, 150)

# Number of images per batch during training/testing
batch_size = 32

# Data augmentation and normalization for training data
train_gen = ImageDataGenerator(
    rescale=1./255,         # Normalize pixel values to [0,1]
    shear_range=0.2,        # Random shear transformation for augmentation
    zoom_range=0.2,         # Random zoom for augmentation
    horizontal_flip=True,   # Randomly flip images horizontally
    validation_split=0.2    # Reserve 20% data for validation
)

# Load training images with augmentation applied, subset as training data
train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',  # Binary labels for cat vs dog
    subset='training'     # Use training split of data
)

# Load validation images with augmentation applied, subset as validation data
val_data = train_gen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'   # Use validation split of data
)

# Build the CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),  # Conv layer with 32 filters
    MaxPooling2D(2, 2),                                               # Downsample by 2x2 max pooling
    Conv2D(64, (3, 3), activation='relu'),                           # Conv layer with 64 filters
    MaxPooling2D(2, 2),
    Flatten(),                                                       # Flatten 3D feature maps to 1D
    Dense(128, activation='relu'),                                  # Fully connected dense layer
    Dropout(0.5),                                                   # Dropout for regularization to prevent overfitting
    Dense(1, activation='sigmoid')                                  # Output layer with sigmoid for binary classification
])

# Compile the model specifying optimizer, loss function, and evaluation metric
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks to stop early if no improvement and save the best model
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),  # Stop if val loss doesn't improve for 3 epochs
    ModelCheckpoint(model_path, save_best_only=True)       # Save only the best model weights
]

# Train the model on training data and validate on validation data
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,          # Train for max 10 epochs
    callbacks=callbacks # Use the callbacks defined above
)

# Plot training and validation accuracy over epochs to visualize performance
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Prepare test data generator with only normalization (no augmentation)
test_gen = ImageDataGenerator(rescale=1./255)

# Load test images without shuffling to keep labels aligned with predictions
test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=1,      # Use batch size 1 for prediction
    class_mode='binary',
    shuffle=False      # Important to disable shuffle for correct evaluation
)

# Load the saved best model from training
model = load_model(model_path)

# Predict probabilities for each test image
preds = model.predict(test_data)

# Convert probabilities to binary predictions (threshold 0.5)
y_pred = (preds > 0.5).astype(int)

# Get the true labels from the test data generator
y_true = test_data.classes

# Print detailed classification metrics
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Print confusion matrix to see true positives, false positives, etc.
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))


# Function to predict a single image from a given file path
def predict_custom_image(image_path):
    # Load and resize image
    img = load_img(image_path, target_size=img_size)
    
    # Convert image to numpy array and normalize pixel values
    img_array = img_to_array(img) / 255.0
    
    # Add batch dimension to array (required by model.predict)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict probability of being a dog (class 1)
    prediction = model.predict(img_array)[0][0]
    
    # Determine label and confidence
    label = "Dog" if prediction > 0.5 else "Cat"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    # Print result with confidence percentage
    print(f"{image_path} → {label} ({confidence*100:.2f}% confidence)")

# Example usage: Predict on some images downloaded from internet folder
predict_custom_image("internet_test/cat1.jpg")
predict_custom_image("internet_test/dog1.jpg")
predict_custom_image("internet_test/dog3.jpg")
