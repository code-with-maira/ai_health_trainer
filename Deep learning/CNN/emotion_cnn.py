
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# SHORTCUTS
# =========================================================

layers = tf.keras.layers
models = tf.keras.models
callbacks = tf.keras.callbacks
preprocessing = tf.keras.preprocessing.image

# =========================================================
# MODELS
# =========================================================

Sequential = models.Sequential

# =========================================================
# LAYERS
# =========================================================

Conv2D = layers.Conv2D
MaxPooling2D = layers.MaxPooling2D
Flatten = layers.Flatten
Dense = layers.Dense
Dropout = layers.Dropout
BatchNormalization = layers.BatchNormalization

# =========================================================
# PREPROCESSING
# =========================================================

ImageDataGenerator = preprocessing.ImageDataGenerator

# =========================================================
# CALLBACKS
# =========================================================

ModelCheckpoint = callbacks.ModelCheckpoint
EarlyStopping = callbacks.EarlyStopping
ReduceLROnPlateau = callbacks.ReduceLROnPlateau


# ── Labels ────────────────────────────────────────────────────────────
# Dataset folder structure:
# dataset/emotion/happy/
# dataset/emotion/sad/
# dataset/emotion/angry/
# dataset/emotion/neutral/
# dataset/emotion/surprised/
# dataset/emotion/fear/
# dataset/emotion/disgust/

LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
NUM_CLASSES = len(LABELS)
IMG_SIZE = (48, 48)     # emotion ke liye 48x48 standard hai (FER dataset)
BATCH_SIZE = 64
EPOCHS = 50

# ── Model banana ──────────────────────────────────────────────────────
# Emotion detection ke liye deeper network chahiye
def build_emotion_model():
    model = Sequential([

        # Block 1
        Conv2D(64, (3,3), activation='relu', padding='same',
               input_shape=(48, 48, 1)),   # grayscale = 1 channel
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 2
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 3
        Conv2D(256, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 4
        Conv2D(512, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Classifier
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

model = build_emotion_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── Data augmentation ─────────────────────────────────────────────────
# Emotion ke liye grayscale use karo — color matter nahi karta
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,       # face mirror ho sakta hai
    zoom_range=0.1,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    'dataset/emotion/',
    target_size=IMG_SIZE,
    color_mode='grayscale',     # grayscale important hai
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    'dataset/emotion/',
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print("Classes:", train_gen.class_indices)

# ── Callbacks ─────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        'weights/emotion_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(           # learning rate automatically kam karta hai
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.000001,
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

# ── Graph ─────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Emotion Model - Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Emotion Model - Loss')
plt.legend()

plt.tight_layout()
plt.savefig('emotion_training_graph.png')
plt.show()

print("Best model saved: weights/emotion_model.h5")

# ── Real-time face emotion detect karo ───────────────────────────────
def predict_emotion_realtime():
    import cv2
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, IMG_SIZE)
            face_roi = face_roi / 255.0
            face_roi = np.expand_dims(face_roi, axis=[0, -1])  # (1,48,48,1)

            pred = model.predict(face_roi, verbose=0)
            label = LABELS[np.argmax(pred)]
            conf  = np.max(pred) * 100

            # Face ke around box
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.0f}%",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# predict_emotion_realtime()  # uncomment karke chalao