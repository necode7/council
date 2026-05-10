from .base import BaseAgent

class Executor(BaseAgent):
    def __init__(self, prompt: str):
        super().__init__(
            name="Executor",
            persona_prompt=prompt,
            role_description="asks 'what do you do Monday morning?'"
        )
