from google.genai import types

FUNCTION_DECLARATIONS = [
    types.FunctionDeclaration(
        name="indicrag_retrieval",
        description=(
            "Retrieve relevant passages from the indexed multilingual corpus using "
            "hybrid BM25 + dense retrieval with cross-encoder reranking. "
            "Always call this first for document questions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query in any of the 10+ supported languages.",
                },
                "expand_query": {
                    "type": "boolean",
                    "description": (
                        "If True, generates 3 query variants before retrieval. "
                        "Use for ambiguous or under-specified queries."
                    ),
                },
            },
            "required": ["query"],
        },
    ),
    types.FunctionDeclaration(
        name="web_search",
        description=(
            "Search the web for information not in the document corpus. "
            "Use for current events, facts outside the corpus, or claim verification."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Web search query."},
                "num_results": {
                    "type": "integer",
                    "description": "Results to return (1-10, default 5).",
                },
            },
            "required": ["query"],
        },
    ),
    types.FunctionDeclaration(
        name="calculate",
        description=(
            "Evaluate a mathematical expression. Use ^ for exponentiation. "
            "Supports sqrt, log, sin, cos, tan, exp, abs."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression, e.g. 'sqrt(144) + 2^10'.",
                },
            },
            "required": ["expression"],
        },
    ),
    types.FunctionDeclaration(
        name="execute_python",
        description=(
            "Execute Python code in a sandboxed environment for data analysis "
            "or string processing. Use print() to produce output."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Use print() for output.",
                },
            },
            "required": ["code"],
        },
    ),
    types.FunctionDeclaration(
        name="arxiv_search",
        description=(
            "Search arXiv for scientific papers by topic, author, or ID. "
            "Returns titles, abstracts, authors, and PDF links. "
            "Use for finding recent research, specific papers, or literature surveys."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — topic, title keywords, or arXiv ID (e.g. '2301.07041').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of papers to return (1-10, default 5).",
                },
                "sort_by": {
                    "type": "string",
                    "description": "Sort order: 'relevance' (default) or 'submitted_date'.",
                },
            },
            "required": ["query"],
        },
    ),
    types.FunctionDeclaration(
        name="open_access_search",
        description=(
            "Search Semantic Scholar for open-access scientific papers across all disciplines. "
            "Returns titles, abstracts, authors, citation counts, and open-access PDF links. "
            "Use for broad academic literature search beyond arXiv, or when citation counts matter."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — topic, title, or research question.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of papers to return (1-10, default 5).",
                },
                "year_range": {
                    "type": "string",
                    "description": "Filter by publication year range, e.g. '2020-2025' or '2023-'. Optional.",
                },
                "open_access_only": {
                    "type": "boolean",
                    "description": "If true, only return papers with open-access PDFs. Default true.",
                },
            },
            "required": ["query"],
        },
    ),
]

TOOLS = types.Tool(function_declarations=FUNCTION_DECLARATIONS)
