# =========================================================
# ai/models/emotion_classifier.py
# =========================================================

import numpy as np
import tensorflow as tf
import os




layers = tf.keras.layers
models = tf.keras.models
callbacks = tf.keras.callbacks
callbacks.EarlyStopping
callbacks.ModelCheckpoint



class EmotionClassifier:

    def __init__(self):

        self.emotions = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral"
        ]

        self.image_size = 48

        self.model = self.build_model()

    # =====================================================
    # BUILD MODEL
    # =====================================================

    def build_model(self):

        model = models.Sequential([
tf.keras.
            layers.Input(
                shape=(48, 48, 1)
            ),

            # -------------------------------------------------
            # Block 1
            # -------------------------------------------------

            layers.Conv2D(
                32,
                (3, 3),
                padding="same",
                activation="relu"
            ),

            layers.BatchNormalization(),

            layers.MaxPooling2D(
                (2, 2)
            ),

            layers.Dropout(0.25),

            # -------------------------------------------------
            # Block 2
            # -------------------------------------------------

            layers.Conv2D(
                64,
                (3, 3),
                padding="same",
                activation="relu"
            ),

            layers.BatchNormalization(),

            layers.MaxPooling2D(
                (2, 2)
            ),

            layers.Dropout(0.25),

            # -------------------------------------------------
            # Block 3
            # -------------------------------------------------

            layers.Conv2D(
                128,
                (3, 3),
                padding="same",
                activation="relu"
            ),

            layers.BatchNormalization(),

            layers.MaxPooling2D(
                (2, 2)
            ),

            layers.Dropout(0.3),

            # -------------------------------------------------
            # Dense Layers
            # -------------------------------------------------

            layers.Flatten(),

            layers.Dense(
                256,
                activation="relu"
            ),

            layers.Dropout(0.5),

            layers.Dense(
                len(self.emotions),
                activation="softmax"
            )
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001
            ),

            loss="sparse_categorical_crossentropy",

            metrics=["accuracy"]
        )

        return model

    # =====================================================
    # TRAIN
    # =====================================================

    def train(
        self,
        X,
        y,
        epochs=30,
        batch_size=64
    ):

        X = np.array(X, dtype=np.float32) / 255.0
        y = np.array(y)

        callbacks = [

            callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True
            ),

            callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=3,
                verbose=1
            ),

            callbacks.ModelCheckpoint(
                "saved_models/best_emotion_model.keras",
                save_best_only=True
            )
        ]

        history = self.model.fit(

            X,
            y,

            validation_split=0.2,

            epochs=epochs,

            batch_size=batch_size,

            callbacks=callbacks,

            verbose=1
        )

        return history.history

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(self, face_image):

        face_image = np.array(
            face_image,
            dtype=np.float32
        )

        if face_image.shape != (48, 48):
            raise ValueError(
                f"Expected image shape (48,48), got {face_image.shape}"
            )

        img = face_image.reshape(
            1,
            48,
            48,
            1
        ) / 255.0

        probs = self.model.predict(
            img,
            verbose=0
        )[0]

        idx = int(np.argmax(probs))

        return {

            "emotion": self.emotions[idx],

            "confidence": round(
                float(probs[idx]),
                3
            ),

            "all_probabilities": {

                emotion: round(float(prob), 3)

                for emotion, prob in zip(
                    self.emotions,
                    probs
                )
            }
        }

    # =====================================================
    # SAVE
    # =====================================================

    def save(
        self,
        path="saved_models/emotion_classifier.keras"
    ):

        os.makedirs(
            "saved_models",
            exist_ok=True
        )

        self.model.save(path)

        print(f"Saved -> {path}")

    # =====================================================
    # LOAD
    # =====================================================

    def load(
        self,
        path="saved_models/emotion_classifier.keras"
    ):

        self.model = tf.keras.models.load_model(path)

        print(f"Loaded <- {path}")


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    clf = EmotionClassifier()

    clf.model.summary()

    # Dummy data
    X = np.random.randint(
        0,
        255,
        (200, 48, 48, 1)
    )

    y = np.random.randint(
        0,
        7,
        200
    )

    clf.train(
        X,
        y,
        epochs=2
    )

    # Fake image
    test_img = np.random.randint(
        0,
        255,
        (48, 48)
    )

    result = clf.predict(test_img)

    print(result)

    clf.save()