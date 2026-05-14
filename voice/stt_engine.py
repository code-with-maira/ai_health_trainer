import speech_recognition as sr

class STTEngine:
    def __init__(self, language="en-US", timeout=5, phrase_limit=10):
        self.recognizer = sr.Recognizer()
        self.language = language
        self.timeout = timeout
        self.phrase_limit = phrase_limit

    def listen(self) -> str | None:
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(
                    source, timeout=self.timeout, phrase_time_limit=self.phrase_limit
                )
                text = self.recognizer.recognize_google(audio, language=self.language)
                return text.lower()
            except sr.WaitTimeoutError:
                return None
            except sr.UnknownValueError:
                return None
            except sr.RequestError as e:
                print(f"STT API error: {e}")
                return None

    def listen_from_file(self, file_path: str) -> str | None:
        with sr.AudioFile(file_path) as source:
            audio = self.recognizer.record(source)
            try:
                return self.recognizer.recognize_google(audio, language=self.language)
            except Exception as e:
                print(f"Error: {e}")
                return None