import tensorflow as tf
import tensorflow as tf

# =========================================================
# SHORTCUTS
# =========================================================

layers = tf.keras.layers
models = tf.keras.models

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
import numpy as np

# ── Model banana ──────────────────────────────────────────
def build_lstm_model(timesteps=30, features=17, num_classes=6):
    # 17 features = body keypoints (x, y coordinates)
    # 30 timesteps = 30 frames ka sequence
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
        Dropout(0.3),

        LSTM(128, return_sequences=True),
        Dropout(0.3),

        LSTM(64),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

model = build_lstm_model()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── Fake data se test (real data nahi hai abhi) ───────────
X_train = np.random.randn(500, 30, 17)   # 500 samples
y_train = tf.keras.utils.to_categorical(
    np.random.randint(0, 6, 500), num_classes=6
)

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
model.save('weights/fatigue_model.h5')
print("LSTM model saved!")