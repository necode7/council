from .contrarian import Contrarian
from .first_principles import FirstPrinciplesThinker
from .expansionist import Expansionist
from .outsider import Outsider
from .executor import Executor
from .chairman import Chairman

def get_advisor_agents(prompt_loader):
    """Return a list of initialized advisor agents."""
    return [
        Contrarian(prompt_loader("advisors/contrarian.txt")),
        FirstPrinciplesThinker(prompt_loader("advisors/first_principles_thinker.txt")),
        Expansionist(prompt_loader("advisors/expansionist.txt")),
        Outsider(prompt_loader("advisors/outsider.txt")),
        Executor(prompt_loader("advisors/executor.txt")),
    ]

def get_chairman_agent(prompt_loader):
    """Return the initialized chairman agent."""
    return Chairman(prompt_loader("chairman/system.txt"))

def get_persona_mappings(prompt_loader):
    """Helper for council.py to maintain backward compatibility."""
    advisors = get_advisor_agents(prompt_loader)
    
    names = [a.name for a in advisors]
    personas = {a.name: a.persona_prompt for a in advisors}
    roles = {a.name: a.role_description for a in advisors}
    
    return names, personas, roles
