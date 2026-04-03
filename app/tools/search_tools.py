"""Tools for search operations."""

from pydantic import BaseModel, Field


class HybridSearchTool(BaseModel):
    """Tool for hybrid search combining vector and keyword search."""
    collection_name: str = Field(..., description="The name of the collection to search in")
    query: str = Field(..., description="The query to search for")


class WebSearchTool(BaseModel):
    """Tool for web search."""
    query: str = Field(..., description="The query to search for")


class SearchTool(BaseModel):
    """Tool for re-searching if needed."""
    query: str = Field(..., description="The query to re-search for.")
