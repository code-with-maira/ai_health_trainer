from fer import FER
import cv2

class EmotionDetector:

    def __init__(self):

        self.detector = FER(mtcnn=False)

    def analyze(self, frame):

        results = self.detector.detect_emotions(frame)

        return results

    def draw(self, frame, results):

        for result in results:

            x, y, w, h = result["box"]

            emotions = result["emotions"]

            emotion = max(emotions, key=emotions.get)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        return frame


if __name__ == "__main__":

    detector = EmotionDetector()

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = detector.analyze(frame)

        frame = detector.draw(frame, results)

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()