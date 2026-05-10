from .base import BaseAgent

class Chairman(BaseAgent):
    def __init__(self, prompt: str):
        super().__init__(
            name="Chairman",
            persona_prompt=prompt,
            role_description="synthesizes a final verdict"
        )
