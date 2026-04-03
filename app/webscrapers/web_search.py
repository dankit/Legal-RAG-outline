import os
from tavily import TavilyClient
from ..tools.search_tools import WebSearchTool

api_key = os.environ.get("TAVILY_API_KEY")
if not api_key:
    raise ValueError("TAVILY_API_KEY environment variable is required. Set it in your .env file.")

client = TavilyClient(api_key=api_key)


def web_search(query: str):
    """Search the web using Tavily and return results as a dict of {url: {title, content, score}}."""
    results = client.search(query, search_depth='advanced')
    result_dict = {
        result['url']: {
            'title': result['title'], 
            'content': result['content'], 
            'similarity_score': result['score']
        } 
        for result in results['results']
    }
    return result_dict
