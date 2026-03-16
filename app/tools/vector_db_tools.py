"""Tools for vector database operations."""

from pydantic import BaseModel, Field


class GetChunkNeighborsTool(BaseModel):
    """Tool for getting neighboring chunks."""
    filename: str = Field(..., description="The name of the file")
    chunkid: int = Field(..., description="The ID of the current chunk")
    k: int = Field(..., description="The window size for how many previous/next chunks to fetch")


class GetChunkByIDTool(BaseModel):
    """Tool for getting a chunk by its ID."""
    id: str = Field(..., description="The ID of the chunk to search for, in format 'filename:chunkid'")
