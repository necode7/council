"""
Base Agent definition for the LLM Council.
"""

class BaseAgent:
    def __init__(self, name: str, persona_prompt: str, role_description: str):
        self.name = name
        self.persona_prompt = persona_prompt
        self.role_description = role_description

    def get_system_prompt(self, suffix: str = "") -> str:
        """Construct the system prompt for this agent."""
        return self.persona_prompt + (suffix if suffix else "")

    def __repr__(self):
        return f"<Agent: {self.name}>"
