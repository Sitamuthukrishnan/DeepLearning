Code:
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = tf.image.resize(x_train, (224, 224)) / 255.0
x_test = tf.image.resize(x_test, (224, 224)) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# Load MobileNet without the top layer
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False # Freeze weights
# Build model
model = Sequential([
base_model,
GlobalAveragePooling2D(),
Dense(128, activation='relu'),
Dense(10, activation='softmax') # 10 classes for CIFAR-10
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train
model.fit(x_train, y_train, validation_split=0.2, epochs=5, batch_size=32)
# Evaluate
loss, acc = model.evaluate(x_test, y
