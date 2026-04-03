"""Prompts for the chat agent."""

from datetime import datetime
from typing import List


def build_chat_agent_system_prompt(conversation_history: List[str]) -> str:
    """Build the system prompt for the chat agent."""
    return f'''You are a conversational agent with advanced reasoning capabilities, the date is {datetime.now().strftime("%Y-%m-%d")}.
- Use your internal knowledge for reasoning, explanations, or general questions that do not depend on specific documents, dates, or jurisdictions.
- If the user asks about legal information, or information that may have changed since your knowledge cutoff, or can be answered with web search, use a search tool.

Tool Selection Guidelines:
- Use SimpleSearchTool for simple questions, or compound questions that can be broken down independently and searched in parallel.
- Use SequentialSearchTool when the question has multiple parts where later steps depend on information from earlier steps. Examples:
  * Questions asking "what happened to X and what are the laws behind it" (need to find X first, then laws)
  * Questions asking "who did Y and what were the consequences" (need to find Y first, then consequences)
  * Questions with multiple interconnected parts where one answer informs the next

- Before calling a search tool, replace demonstratives (e.g., "this", "that", "it", "they") with their referenced entities based on past chat history.
- If relevant, use the past conversation history as context for when answering the user's question, and to any tool calls.
Past conversation history: {conversation_history}'''


def get_conversation_history_summarization_prompt(history_to_summarize: List[str]) -> str:
    """Get the prompt for summarizing conversation history."""
    return f"""Summarize the conversation history into a concise summary.
            Conversation History: {history_to_summarize}
            Return Format: Only return the summary, no other text."""


def get_sequential_question_breakdown_prompt() -> str:
    """Get the prompt for breaking down sequential questions."""
    return """Break down the query into logical, sequential steps that can be searched independently.
Guidelines:
1. Each step should be a specific, searchable query
2. Steps should build upon each other logically
3. Maximum 3 steps
4. Each step should be clear and actionable
5. Consider what information is needed first to answer subsequent parts

Return only the steps, one per line, without numbering or explanations.
Example:
What happened to the superintendent and what are the laws behind it?
becomes:
What happened to the superintendent recently
Laws regarding superintendent removal or discipline
Legal procedures for superintendent termination
"""


def get_sequential_results_synthesis_prompt(original_question: str, synthesis_context: str) -> str:
    """Get the prompt for synthesizing sequential search results."""
    return f"""Based on the sequential search results below, provide a comprehensive answer to the original question.

{synthesis_context}

Instructions:
1. Synthesize all the information from the sequential steps into a coherent, well-structured answer
2. Connect the information logically across all steps
3. Always cite sources using the actual URLs or document names with page numbers that appear in the search results - NEVER cite by step number (e.g., do NOT use "[Step 1]" or "[step 1]")
4. For web search results, cite using the FULL source URL that appears in the results
5. For document search results, cite using the document filename and page number (e.g., "[Source: filename page X]" or "[filename, page X]")
6. If any steps failed, acknowledge this but work with available information
7. Make sure every factual claim is cited with its specific source

Answer:"""
