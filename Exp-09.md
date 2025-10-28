CODE:
import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
# Step 1: Load and preprocess dataset
# Assume directory structure: dataset/train/with_mask, dataset/train/without_mask, etc.
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, shear_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
'dataset/train',
target_size=(100, 100),
batch_size=32,
class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
'dataset/val',
target_size=(100, 100),
batch_size=32,
class_mode='binary'
)
# Step 2: Build CNN model
model = Sequential([
Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
MaxPooling2D(2, 2),
Conv2D(64, (3, 3), activation='relu'),
MaxPooling2D(2, 2),
Flatten(),
Dense(128, activation='relu'),
Dropout(0.5),
Dense(1, activation='sigmoid')
])
# Step 3: Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=val_generator, epochs=10)
# Save the model
model.save('face_mask_detector.h5')
Test the Model on New Images
from keras.models import load_model
from keras.preprocessing import image
model = load_model('face_mask_detector.h5')
img = image.load_img('test_image.jpg', target_size=(100, 100))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)
if prediction[0][0] > 0.5:
print("Without Mask")
else:
print("With Mask")
