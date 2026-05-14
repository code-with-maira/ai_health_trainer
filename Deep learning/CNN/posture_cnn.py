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
BatchNormalization = layers.BatchNormalization

ImageDataGenerator = preprocessing.ImageDataGenerator
callbacks = tf.keras.callbacks
callbacks.EarlyStopping = callbacks.EarlyStopping
callbacks.ModelCheckpoint = callbacks.ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

# ── Labels ────────────────────────────────────────────────────────────
# Tumhare dataset mein yeh folders honge:
# dataset/posture/good/
# dataset/posture/bad_hunched/
# dataset/posture/bad_leaning/

LABELS = ['good', 'bad_hunched', 'bad_leaning']
NUM_CLASSES = len(LABELS)
IMG_SIZE = (128, 128)   # posture ke liye thoda bada size better hai
BATCH_SIZE = 32
EPOCHS = 30

# ── Model banana ──────────────────────────────────────────────────────
def build_posture_model():
    model = Sequential([

        # Block 1
        Conv2D(32, (3,3), activation='relu', padding='same',
               input_shape=(128, 128, 3)),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Classifier
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

model = build_posture_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── Data augmentation ─────────────────────────────────────────────────
# Posture ke liye horizontal flip mat karo — left/right matter karta hai
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,        # thodi si rotation
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    'dataset/posture/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    'dataset/posture/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print("Classes:", train_gen.class_indices)

# ── Callbacks ─────────────────────────────────────────────────────────
callbacks = [
    callbacks.ModelCheckpoint(
        'weights/posture_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
]

# ── Training ──────────────────────────────────────────────────────────
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)

# ── Accuracy graph ────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Posture Model - Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Posture Model - Loss')
plt.legend()

plt.tight_layout()
plt.savefig('posture_training_graph.png')
plt.show()

print("Best model saved: weights/posture_model.h5")

# ── Single image test ─────────────────────────────────────────────────
def predict_posture(image_path):
    preprocessing = tf.keras.preprocessing.image
    img = preprocessing.load_img(image_path, target_size=IMG_SIZE)
    img_array = preprocessing.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    label = LABELS[np.argmax(pred)]
    confidence = np.max(pred) * 100

    print(f"Posture: {label}  |  Confidence: {confidence:.1f}%")
    return label

# predict_posture('test_image.jpg')  # uncomment karke test karo