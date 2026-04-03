"""Tools for the chat agent."""

from pydantic import BaseModel, Field


class SimpleSearchTool(BaseModel):
    """Tool for searching for a single, straightforward question that can be answered with one search."""
    query: str = Field(..., description="The query to search for")
    context: str = Field(..., description="Additional context for the search query, only provide if needed to answer the question.")


class SequentialSearchTool(BaseModel):
    """Tool for answering complex multi-part questions where later steps depend on answers from earlier steps. Use this when the question requires breaking into sequential steps that build upon each other."""
    question: str = Field(..., description="The complex question to break down into sequential steps")
    reasoning: str = Field(default="", description="Brief reasoning for why this needs sequential processing (optional)")
    context: str = Field(..., description="Additional context for the search query, only provide if needed to answer the question.")


class StepSearchTool(BaseModel):
    """Tool for searching for a specific step in a sequential process."""
    step_query: str = Field(..., description="The specific query for this step")
    step_number: int = Field(..., description="The step number in the sequence")
    context: str = Field(..., description="Context from previous steps")
