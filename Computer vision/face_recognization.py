import cv2
import numpy as np

class FaceRecognizer:

    def __init__(self):

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            'haarcascade_frontalface_default.xml'
        )

    def recognize(self, frame: np.ndarray) -> list:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        results = []

        for (x, y, w, h) in faces:

            results.append({
                "name": "Person",
                "box": (x, y, x+w, y+h)
            })

        return results

    def draw(self, frame: np.ndarray, results: list):

        for r in results:

            l, t, ri, b = r["box"]

            cv2.rectangle(
                frame,
                (l, t),
                (ri, b),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                r["name"],
                (l, t - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        return frame


if __name__ == "__main__":

    recognizer = FaceRecognizer()

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = recognizer.recognize(frame)

        frame = recognizer.draw(frame, results)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()