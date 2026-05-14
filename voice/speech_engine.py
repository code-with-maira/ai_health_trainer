from voice.stt_engine import STTEngine
from voice.tts_engine import TTSEngine
from voice.voice_commands import VoiceCommandHandler

class SpeechEngine:
    def __init__(self):
        self.stt = STTEngine()
        self.tts = TTSEngine()
        self.commands = VoiceCommandHandler()

    def listen_and_respond(self):
        print("Listening...")
        text = self.stt.listen()
        if not text:
            return
        print(f"Heard: {text}")
        response = self.commands.handle(text)
        if response:
            print(f"Response: {response}")
            self.tts.speak(response)

    def speak(self, text: str):
        self.tts.speak(text)

    def listen(self) -> str:
        return self.stt.listen()