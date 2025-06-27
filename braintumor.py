

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import VGG16
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import cv2  # For image segmentation

plt.style.use('dark_background')

# Update encoder for 4 classes: glioma, meningioma, pituitary, and no tumor
encoder = OneHotEncoder()
encoder.fit([[0], [1], [2], [3]])  # 0 - Glioma, 1 - Meningioma, 2 - Pituitary, 3 - No Tumor

data, result = [], []

# Define directories
glioma_dir = r"C:\Users\91936\Desktop\brain_tumor\pythonProject8\datasets\training_sets\Training\glioma_tumor"
meningioma_dir = r"C:\Users\91936\Desktop\brain_tumor\pythonProject8\datasets\training_sets\Training\meningioma_tumor"
pituitary_dir = r"C:\Users\91936\Desktop\brain_tumor\pythonProject8\datasets\training_sets\Training\pituitary_tumor"
no_tumor_dir = r"C:\Users\91936\Desktop\brain_tumor\pythonProject8\datasets\training_sets\Training\no_tumor"

# Load images and labels
for label, directory in enumerate([glioma_dir, meningioma_dir, pituitary_dir, no_tumor_dir]):
    for r, d, f in os.walk(directory):
        for file in f:
            if '.jpg' in file:
                img = Image.open(os.path.join(r, file))
                img = img.resize((128, 128))
                img = np.array(img)
                if img.shape == (128, 128, 3):
                    data.append(img)
                    result.append(encoder.transform([[label]]).toarray()[0])

# Convert lists to numpy arrays and normalize pixel values
data = np.array(data) / 255.0
result = np.array(result)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data, result, test_size=0.2, shuffle=True, random_state=0)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
datagen.fit(x_train)

# Load the VGG16 model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
for layer in base_model.layers[-4:]:
    layer.trainable = True  # Unfreeze the last 4 layers for fine-tuning

# Define the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 classes
])

# Compile the Model with additional metrics
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

# Early Stopping and Learning Rate Scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)

# Train the Model
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)`

# Evaluate the Model
loss, accuracy, precision, recall = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")

# Generate a classification report
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor", "No Tumor"]))

# Confusion Matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))

# Plot Training and Validation Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()

# Plot Training and Validation Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Segmentation using OpenCV
def segment_tumor(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Tumor Segmentation')
    plt.show()

# Load a New Image for Prediction
img_path = r"C:\Users\91936\Desktop\brain_tumor\pythonProject8\datasets\training_sets\Training\meningioma_tumor\m (5).jpg"
img = Image.open(img_path)
x = np.array(img.resize((128, 128))) / 255.0  # Normalize pixel values
x = x.reshape(1, 128, 128, 3)  # Reshape correctly

res = model.predict(x)  # Predict for the new image
classification = np.argmax(res[0])  # Get the predicted class
print(f'{res[0][classification]*100:.2f}% Confidence This is {["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor", "No Tumor"][classification]}')

# Segment the tumor from the image
segment_tumor(img_path)

# Predict tumor severity based on confidence
severity = "Severe" if res[0][classification] > 0.8 else "Moderate" if res[0][classification] > 0.5 else "Mild"
print(f"Predicted Severity: {severity}")
