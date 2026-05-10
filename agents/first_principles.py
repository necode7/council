from .base import BaseAgent

class FirstPrinciplesThinker(BaseAgent):
    def __init__(self, prompt: str):
        super().__init__(
            name="First Principles Thinker",
            persona_prompt=prompt,
            role_description="rebuilds the problem from scratch"
        )
