# =========================================================
# MOTION PREDICTION MODEL (LSTM)
# =========================================================

import os
import numpy as np
import tensorflow as tf

# =========================================================
# SHORTCUTS
# =========================================================

layers = tf.keras.layers
models = tf.keras.models
callbacks = tf.keras.callbacks

# =========================================================
# MODELS
# =========================================================

Sequential = models.Sequential

# =========================================================
# LAYERS
# =========================================================

LSTM = layers.LSTM
Dense = layers.Dense
Dropout = layers.Dropout

# =========================================================
# CALLBACKS
# =========================================================

EarlyStopping = callbacks.EarlyStopping
ModelCheckpoint = callbacks.ModelCheckpoint

# =========================================================
# BUILD LSTM MODEL
# =========================================================

def build_lstm_model(sequence_length=30,
                     feature_dim=99,
                     num_classes=5):

    model = Sequential([

        LSTM(
            128,
            return_sequences=True,
            input_shape=(sequence_length, feature_dim)
        ),

        Dropout(0.3),

        LSTM(64),

        Dropout(0.3),

        Dense(
            64,
            activation='relu'
        ),

        Dense(
            num_classes,
            activation='softmax'
        )
    ])

    return model


# =========================================================
# BUILD MODEL
# =========================================================

model = build_lstm_model()

# =========================================================
# COMPILE MODEL
# =========================================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================================================
# MODEL SUMMARY
# =========================================================

model.summary()

# =========================================================
# DUMMY TRAINING DATA
# Replace with real sequence dataset
# =========================================================

X = np.random.rand(500, 30, 99)

y = tf.keras.utils.to_categorical(
    np.random.randint(0, 5, 500),
    num_classes=5
)

# =========================================================
# CALLBACKS
# =========================================================

os.makedirs("weights", exist_ok=True)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "weights/best_motion_model.keras",
    save_best_only=True
)

# =========================================================
# TRAIN MODEL
# =========================================================

history = model.fit(

    X,
    y,

    validation_split=0.2,

    epochs=30,

    batch_size=32,

    callbacks=[early_stop, checkpoint]
)

# =========================================================
# SAVE FINAL MODEL
# =========================================================

MODEL_PATH = "weights/motion_prediction_model.keras"

model.save(MODEL_PATH)

print(f"\nModel Saved Successfully → {MODEL_PATH}")