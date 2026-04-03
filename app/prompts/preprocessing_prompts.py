"""Prompts for preprocessing operations."""


def get_pdf_chunking_prompt(chunk_size: int) -> str:
    """Get the prompt for PDF chunking."""
    return f"""You are a PDF text chunking agent.
        Task:
        - Split the page into semantically meaningful chunks, each < {chunk_size} characters.
        - Carry over cut-off, incomplete semantic chunks if it continues over to the next page.
        - Prepend carry-over text from the previous page to the first chunk on this page.
        - If the page has no meaningful content, return metadata with an empty "text" field.
        - Extract structural elements (e.g., headers, footers, chapters) as metadata.
        Input:
        -Previous page carry over: '''{{carry_over_text}}'''
        -Current page: '''{{input_text}}'''
        Goal:
        - Return a list of chunks (text + metadata), and carry-over text.
        - Chunks should preserve the original text closely (minimal whitespace cleanup only).
        - Include concise, primitive key-value metadata for downstream filtering (e.g., laws, sections, etc.).
        Output Format (JSON string, minified):
        -'{{"chunks": [{{"text": "string", "metadata": {{"topic": "string", "section": "string"}}}}], "text_carry_over": "string"}}'"""
