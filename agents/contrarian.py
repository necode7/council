from .base import BaseAgent

class Contrarian(BaseAgent):
    def __init__(self, prompt: str):
        super().__init__(
            name="Contrarian",
            persona_prompt=prompt,
            role_description="stress-tests for failure modes"
        )
