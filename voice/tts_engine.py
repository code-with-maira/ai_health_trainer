import pyttsx3
import threading

class TTSEngine:
    def __init__(self, rate=175, volume=1.0, voice_index=0):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)
        voices = self.engine.getProperty("voices")
        if voices:
            self.engine.setProperty("voice", voices[voice_index].id)
        self._lock = threading.Lock()

    def speak(self, text: str):
        with self._lock:
            self.engine.say(text)
            self.engine.runAndWait()

    def speak_async(self, text: str):
        t = threading.Thread(target=self.speak, args=(text,))
        t.daemon = True
        t.start()

    def set_rate(self, rate: int):
        self.engine.setProperty("rate", rate)