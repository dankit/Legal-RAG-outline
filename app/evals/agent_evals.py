from ..agents.chat_agent import ChatAgent
from ..search.search_agent import SearchAgent


class AgentEvals:
    """Evaluation harness for chat and search agents."""

    def __init__(self):
        self.chat_agent = ChatAgent()
        self.search_agent = SearchAgent()
        self.query_histories = []
