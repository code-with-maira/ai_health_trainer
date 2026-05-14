import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models
preprocessing = tf.keras.preprocessing.image

Sequential = models.Sequential

Conv2D = layers.Conv2D
MaxPooling2D = layers.MaxPooling2D
Flatten = layers.Flatten
Dense = layers.Dense
Dropout = layers.Dropout

ImageDataGenerator = preprocessing.ImageDataGenerator
import numpy as np

# ── Model banana ──────────────────────────────────────────
def build_cnn_model(input_shape=(64, 64, 3), num_classes=5):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # num_classes = tumhari exercises
    ])
    return model

# ── Model compile karo ────────────────────────────────────
model = build_cnn_model(num_classes=5)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── Data load karo ────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    'DATASETS/exercise_recognization/',   # apna folder path
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    'dataset/exercise_recognization/',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ── Training ──────────────────────────────────────────────
history = model.fit(
    train_gen,
    epochs=25,
    validation_data=val_gen
)

# ── Weights save karo ─────────────────────────────────────
model.save('weights/exercise_model.h5')
print("Model saved!")

# ── Colab se download karo ────────────────────────────────
#from google.colab import files
#files.download('weights/exercise_model.h5')#