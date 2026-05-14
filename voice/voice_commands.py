import re
from typing import Callable

class VoiceCommandHandler:
    def __init__(self):
        self.commands: list[tuple[str, Callable]] = []
        self._register_defaults()

    def register(self, pattern: str, handler: Callable):
        self.commands.append((pattern.lower(), handler))

    def _register_defaults(self):
        self.register(r"(hello|hi|hey)", lambda m: "Hello! How can I help?")
        self.register(r"what time is it", lambda m: self._get_time())
        self.register(r"stop|exit|quit", lambda m: self._stop())

    def handle(self, text: str) -> str | None:
        text = text.lower().strip()
        for pattern, handler in self.commands:
            match = re.search(pattern, text)
            if match:
                return handler(match)
        return f"Command not recognized: '{text}'"

    def _get_time(self) -> str:
        from datetime import datetime
        return f"It is {datetime.now().strftime('%I:%M %p')}"

    def _stop(self) -> str:
        return "Shutting down."