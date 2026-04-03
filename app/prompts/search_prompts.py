"""Prompts for search-related operations."""


def get_query_expansion_prompt(is_web_search: bool) -> str:
    """Get the prompt for query expansion."""
    if is_web_search:
        return """Rephrase the user's query to maximize web search relevance as the current one did not provide relevant results."""
    else:
        return """Preprocess the user's query to maximize search relevance:
-Expand all abbreviations or acronyms to their full canonical form (e.g. "DUI" = "drinking under the influence").
-Normalize and rephrase each query to include relevant, unambiguous, high-signal keywords suitable for information retrieval.
Return Format: Output a list of UP TO three queries split by new lines.
e.g. "DUI on school ground laws" becomes "Drinking under the influence laws" and "enhanced school ground penalties for alcohol".
e.g. "What is the zoning law for industrial land and who oversees it?" becomes "Zoning laws for industrial land" and "industrial zoning laws authority"""


def get_search_method_selection_prompt(collections: list) -> str:
    """Get the prompt for selecting search method."""
    return f"""Route the user's query to the best tool to answer the query.
If the user's query is about general laws, legal procedures, or court rules that can be answered with a legal document from the collections, use hybrid search.
If the user's query is about current events, news, specific caselaw, or non-legal information, use web search.
If using hybrid search, only pick ONE collection at most from the collections available. If no relevant collections exist, use web search.
The collections available for hybrid search are: {collections}."""


def get_result_synthesis_prompt(query: str, can_requery: bool) -> str:
    """Get the prompt for synthesizing search results."""
    base_prompt = f"""Answer the user's query based on the summaries of the subqueries it was broken down into. Always cite either the document with page number, or FULL web source url in the answer.
The user's query is: {query}."""
    
    if can_requery:
        return base_prompt + "\nIf there are irrelevant or low-confidence answers for decisive subqueries to the main query, use the search tool to gather more information."
    else:
        return base_prompt + "\nIf there are irrelevant or low-confidence answers for decisive subqueries, try a best attempt to answer but clearly articulate uncertainty."


def get_document_summarization_prompt(query: str, prev_tool_calls: list) -> str:
    """Get the prompt for summarizing documents."""
    return f"""Summarize ONLY the most relevant documents in context of the user's query and return a concise answer. 
Cite answer with ONLY relevant page numbers and sections as sources.
The documents are in dictionary format {{'id' : 'document_text'}}.
Example return format: "Answer to question [Source: filename page X, section Y]"
If more context or metadata is needed, use tools to fetch more information and repeat the process.
Before calling the tool, ensure that it has not already been called. If it has been called, try to prevent duplicate work by modifying the tool call arguments.
Query: {query}
Previous tool calls: {prev_tool_calls}"""


def get_web_results_summarization_prompt(query: str) -> str:
    """Get the prompt for summarizing web search results."""
    return f"""Summarize the web results in context of the user's query and return a concise answer. Cite facts with FULL source urls. The inputs are in dictionary format {{'url' : 'result_text'}}.
Return the summary in a single paragraph, and remove any extra noise or formatting.
Query: {query}"""
