from .base import BaseAgent

class Outsider(BaseAgent):
    def __init__(self, prompt: str):
        super().__init__(
            name="Outsider",
            persona_prompt=prompt,
            role_description="catches the curse of knowledge"
        )
