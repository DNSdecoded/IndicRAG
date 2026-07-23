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
                "year_from": {
                    "type": "integer",
                    "description": (
                        "Only return passages from papers published in this year or "
                        "later (e.g. 2020). Optional; use when the query restricts by year."
                    ),
                },
                "year_to": {
                    "type": "integer",
                    "description": (
                        "Only return passages from papers published in this year or "
                        "earlier. Optional."
                    ),
                },
                "tags": {
                    "type": "string",
                    "description": (
                        "Comma-separated tags to filter retrieval (e.g. 'transformer,efficiency'). "
                        "Only return passages from papers tagged with at least one of these tags. Optional."
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
                "year_from": {
                    "type": "integer",
                    "description": (
                        "Only return papers submitted in this year or later (e.g. 2022). "
                        "Optional; use when the query restricts by year."
                    ),
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
    types.FunctionDeclaration(
        name="create_watch",
        description=(
            "Create a scheduled topic watch that periodically searches for new papers "
            "on a topic, ingests them, and builds a digest. Use when the user says "
            "'keep me updated on X', 'monitor Y', or 'watch for new papers about Z'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic to watch (e.g. 'transformer antenna optimization').",
                },
                "cadence": {
                    "type": "string",
                    "description": "How often to check: 'daily', 'weekly', or 'monthly'. Default 'weekly'.",
                },
                "language": {
                    "type": "string",
                    "description": "Language for the digest (default 'en').",
                },
            },
            "required": ["topic"],
        },
    ),
    types.FunctionDeclaration(
        name="generate_report",
        description=(
            "Generate a structured literature-review report on a topic from the indexed "
            "corpus. Use when the user asks for a 'review', 'report', 'survey', or "
            "'summary of the literature'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic for the literature review.",
                },
                "language": {
                    "type": "string",
                    "description": "Language for the report (default 'en').",
                },
            },
            "required": ["topic"],
        },
    ),
]

TOOLS = types.Tool(function_declarations=FUNCTION_DECLARATIONS)
