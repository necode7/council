from .base import BaseAgent

class Expansionist(BaseAgent):
    def __init__(self, prompt: str):
        super().__init__(
            name="Expansionist",
            persona_prompt=prompt,
            role_description="hunts for hidden upside"
        )
