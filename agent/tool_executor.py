import ast
import time
import re
import logging
import json
import os
import urllib.request
import urllib.parse
import urllib.error

import subprocess
import sys
import tempfile

import numexpr
import arxiv
from tavily import TavilyClient

import rag
import config

logger = logging.getLogger(__name__)
_tavily = None
_S2_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")


def _get_tavily():
    global _tavily
    if _tavily is None:
        key = os.environ.get("TAVILY_API_KEY")
        if not key:
            raise ValueError("TAVILY_API_KEY not configured. Web search unavailable.")
        _tavily = TavilyClient(api_key=key)
    return _tavily


def _expand_query_variants(query: str) -> list[str]:
    from google.genai import types

    prompt = (
        f'Generate 3 alternative phrasings for this search query.\n'
        f'Return JSON only: {{"variants": ["...", "...", "..."]}}\n'
        f'Query: {query}'
    )
    try:
        resp = rag.generate_with_failover(
            model=config.LLM_MODEL_NAME,
            contents=prompt,
            gen_config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=256,
                system_instruction="Generate alternative search phrasings that preserve the original query's semantic meaning. Do not add new topics or narrow the scope.",
            ),
        )
        clean = re.sub(r"```(?:json)?|```", "", resp.text or "").strip()
        return json.loads(clean)["variants"]
    except Exception as e:
        logger.debug(f"Query expansion failed, using original: {e}")
        return [query]


def execute_indicrag(query: str, expand_query: bool = False) -> dict:
    import hashlib
    _MIN_EXPAND_WORDS = 4
    should_expand = expand_query and len(query.split()) >= _MIN_EXPAND_WORDS

    if should_expand:
        variants = _expand_query_variants(query)
        passages, seen = [], set()
        for q in [query] + variants:
            result = rag.retrieve_context(q)
            for chunk, meta in zip(result["chunks"], result["metadatas"]):
                key = hashlib.sha256(chunk.encode()).hexdigest()
                if key not in seen:
                    passages.append({"text": chunk, **meta})
                    seen.add(key)
        return {"passages": passages[: config.MAX_CONTEXT_CHUNKS]}
    else:
        result = rag.retrieve_context(query)
        passages = [
            {"text": chunk, **meta}
            for chunk, meta in zip(result["chunks"], result["metadatas"])
        ]
        return {"passages": passages}


def execute_web_search(query: str, num_results: int = 5) -> dict:
    results = _get_tavily().search(
        query=query,
        max_results=min(num_results, 10),
        search_depth="basic",
        include_raw_content=False,
    )
    return {
        "passages": [
            {"text": r["content"], "title": r["title"], "source": r["url"]}
            for r in results.get("results", [])
        ]
    }


_SAFE_MATH_NAMES = re.compile(
    r'\b(sqrt|log|log10|log2|sin|cos|tan|asin|acos|atan|atan2|exp|abs|pi|e|inf)\b'
)


def execute_calculate(expression: str) -> dict:
    cleaned = expression.replace("^", "**").strip()
    stripped = _SAFE_MATH_NAMES.sub("", cleaned)
    if re.search(r'[a-zA-Z_]', stripped):
        return {"text": "Invalid expression: only numeric operations allowed.", "source": "calculator"}
    try:
        result = float(numexpr.evaluate(cleaned))
        return {"text": f"Result: {result}", "source": "calculator"}
    except Exception as e:
        return {"text": f"Calculation error: {e}", "source": "calculator"}


_SANDBOX_TIMEOUT = 10

_ALLOWED_MODULES = frozenset({
    "math", "statistics", "decimal", "fractions",
    "collections", "itertools", "functools",
    "re", "json", "datetime", "random", "string",
})

_DANGEROUS_NAMES = frozenset({
    "eval", "exec", "compile", "execfile",
    "__import__", "breakpoint", "exit", "quit",
    "open", "input", "help", "credits", "license",
    "getattr", "setattr", "delattr", "vars", "dir",
    "globals", "locals", "memoryview", "type",
})

_DUNDER_IN_STRING = re.compile(r"__\w+__")
_FORMAT_METHODS = frozenset({"format", "format_map"})


def _validate_ast(code: str) -> str | None:
    """Parse code and validate AST. Returns error message or None if safe."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            else:
                names = [node.module.split(".")[0]] if node.module else []
            for name in names:
                if name not in _ALLOWED_MODULES:
                    return f"Import '{name}' is not allowed. Allowed: {', '.join(sorted(_ALLOWED_MODULES))}"

        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            return f"Access to dunder attribute '{node.attr}' is not allowed"

        # Block dunder names used as variables (e.g. __builtins__)
        if isinstance(node, ast.Name) and node.id.startswith("__") and node.id.endswith("__"):
            return f"Access to dunder name '{node.id}' is not allowed"

        # Block subscript access to dunder names in strings (e.g. x['__import__'])
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str) and _DUNDER_IN_STRING.search(node.slice.value):
                return "Subscript access with dunder key is not allowed"

        # Block format-string sandbox escapes: "{0.__class__}".format(x)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _DUNDER_IN_STRING.search(node.value):
                return "String constants must not contain dunder patterns"

        # Block .format() / .format_map() calls (format-string injection vector)
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in _FORMAT_METHODS:
                return f"Call to '.{func.attr}()' is not allowed in sandbox"
            if isinstance(func, ast.Name) and func.id in _DANGEROUS_NAMES:
                return f"Call to '{func.id}()' is not allowed"
            if isinstance(func, ast.Attribute) and func.attr in _DANGEROUS_NAMES:
                return f"Call to '.{func.attr}()' is not allowed"

        # Block %-formatting on strings (another format-string vector)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                if _DUNDER_IN_STRING.search(node.left.value):
                    return "%-formatting with dunder patterns is not allowed"

    return None


_SANDBOX_WRAPPER = '''\
import sys as _sys
for mod in list(_sys.modules.keys()):
    if mod not in ["sys", "builtins", "_imp", "_thread", "_warnings", "_weakref", "encodings", "codecs", "io", "abc", "os", "site"]:
        try: del _sys.modules[mod]
        except KeyError: pass
_SAFE_BUILTINS = {
    "abs", "all", "any", "bin", "bool", "bytes", "chr", "complex",
    "dict", "divmod", "enumerate", "filter", "float", "frozenset",
    "hash", "hex", "int", "isinstance", "issubclass", "iter", "len",
    "list", "map", "max", "min", "next", "oct", "ord", "pow", "print",
    "range", "repr", "reversed", "round", "set", "slice", "sorted",
    "str", "sum", "tuple", "zip",
}
_ALLOWED_IMPORTS = {
    "math", "statistics", "decimal", "fractions",
    "collections", "itertools", "functools",
    "re", "json", "datetime", "random", "string",
}
import builtins as _b
_safe = {k: getattr(_b, k) for k in _SAFE_BUILTINS if hasattr(_b, k)}
_safe["__build_class__"] = _b.__build_class__
_real_import = _b.__import__
def _restricted_import(name, *args, **kwargs):
    top = name.split(".")[0]
    if top not in _ALLOWED_IMPORTS:
        raise ImportError(f"Import '{top}' is not allowed")
    return _real_import(name, *args, **kwargs)
_safe["__import__"] = _restricted_import
exec(
    compile(open(_sys.argv[1]).read(), _sys.argv[1], "exec"),
    {"__builtins__": _safe, "__name__": "__main__"},
)
'''


def execute_python(code: str) -> dict:
    error = _validate_ast(code)
    if error:
        return {"text": f"Blocked: {error}", "source": "code_executor"}

    tmp_path = None
    wrapper_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            tmp_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='_wrapper.py', delete=False, encoding='utf-8') as f:
            f.write(_SANDBOX_WRAPPER)
            wrapper_path = f.name
        env = {
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "PYTHONDONTWRITEBYTECODE": "1"
        }
        result = subprocess.run(
            [sys.executable, "-I", "-u", wrapper_path, tmp_path],
            capture_output=True, text=True, timeout=_SANDBOX_TIMEOUT,
            env=env,
            cwd=tempfile.gettempdir(),
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            err = result.stderr.strip()
            return {"text": f"Runtime error: {err}\n{output}".strip(), "source": "code_executor"}
        return {"text": output or "(no output)", "source": "code_executor"}
    except subprocess.TimeoutExpired:
        return {"text": f"Execution timed out after {_SANDBOX_TIMEOUT}s.", "source": "code_executor"}
    except Exception as e:
        return {"text": f"Sandbox error: {e}", "source": "code_executor"}
    finally:
        for p in (tmp_path, wrapper_path):
            if p:
                try:
                    os.unlink(p)
                except Exception:
                    pass


_ARXIV_TIMEOUT = 20  # seconds


def execute_arxiv_search(
    query: str,
    max_results: int = 5,
    sort_by: str = "relevance",
    year_from: int | None = None,
) -> dict:
    import datetime
    import concurrent.futures as _cf

    sort = (arxiv.SortCriterion.Relevance
            if sort_by != "submitted_date"
            else arxiv.SortCriterion.SubmittedDate)
    client = arxiv.Client(num_retries=1)
    search = arxiv.Search(
        query=query,
        max_results=min(max_results * 3, 30),  # over-fetch to absorb date filtering
        sort_by=sort,
    )
    cutoff = (
        datetime.datetime(year_from, 1, 1, tzinfo=datetime.timezone.utc)
        if year_from else None
    )

    def _fetch() -> list:
        results = []
        for paper in client.results(search):
            if cutoff and paper.published.replace(tzinfo=datetime.timezone.utc) < cutoff:
                continue
            authors = ", ".join(a.name for a in paper.authors[:5])
            if len(paper.authors) > 5:
                authors += f" (+{len(paper.authors) - 5} more)"
            text = (
                f"Title: {paper.title}\n"
                f"Authors: {authors}\n"
                f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"Abstract: {paper.summary[:600]}"
            )
            results.append({
                "text": text,
                "title": paper.title,
                "source": paper.entry_id,
                "section": "arxiv",
                "pdf_url": paper.pdf_url,
                "arxiv_id": paper.entry_id.split("/")[-1],
            })
            if len(results) >= max_results:
                break
        return results

    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as pool:
            passages = pool.submit(_fetch).result(timeout=_ARXIV_TIMEOUT)
    except _cf.TimeoutError:
        logger.warning(f"[ArxivSearch] Timed out after {_ARXIV_TIMEOUT}s")
        passages = []
    return {"passages": passages}


_S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_FIELDS = "title,abstract,authors,year,citationCount,openAccessPdf,externalIds,url"

_OPENALEX_API = "https://api.openalex.org/works"


def _parse_s2_papers(data: dict) -> list[dict]:
    passages = []
    for paper in data.get("data", []):
        authors = ", ".join(a.get("name", "") for a in (paper.get("authors") or [])[:5])
        if len(paper.get("authors") or []) > 5:
            authors += f" (+{len(paper['authors']) - 5} more)"
        pdf_url = ""
        oa = paper.get("openAccessPdf")
        if oa and isinstance(oa, dict):
            pdf_url = oa.get("url", "")
        abstract = (paper.get("abstract") or "No abstract available.")[:600]
        text = (
            f"Title: {paper.get('title', 'Unknown')}\n"
            f"Authors: {authors}\n"
            f"Year: {paper.get('year', 'N/A')}\n"
            f"Citations: {paper.get('citationCount', 0)}\n"
            f"Abstract: {abstract}"
        )
        passages.append({
            "text": text,
            "title": paper.get("title", "Unknown"),
            "source": paper.get("url", ""),
            "section": "semantic_scholar",
            "pdf_url": pdf_url,
            "year": str(paper.get("year", "")),
            "authors": authors,
            "citations": paper.get("citationCount", 0),
        })
    return passages


def _fetch_openalex(query: str, max_results: int, year_range: str, open_access_only: bool) -> list[dict]:
    params = {"search": query, "per_page": min(max_results, 10)}
    if open_access_only:
        params["filter"] = "open_access.is_oa:true"
    if year_range:
        parts = year_range.split("-")
        if len(parts) == 2:
            f = f"from_publication_date:{parts[0]}-01-01"
            if parts[1]:
                f += f",to_publication_date:{parts[1]}-12-31"
            params["filter"] = params.get("filter", "") + ("," if "filter" in params else "") + f

    url = f"{_OPENALEX_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "mailto:indicrag@example.com"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())

    passages = []
    for work in data.get("results", []):
        raw_authors = work.get("authorships", [])
        authors = ", ".join(
            a.get("author", {}).get("display_name", "") for a in raw_authors[:5]
        )
        if len(raw_authors) > 5:
            authors += f" (+{len(raw_authors) - 5} more)"
        pdf_url = ""
        oa_info = work.get("open_access", {})
        if oa_info.get("oa_url"):
            pdf_url = oa_info["oa_url"]
        best_loc = work.get("best_oa_location") or {}
        if not pdf_url and best_loc.get("pdf_url"):
            pdf_url = best_loc["pdf_url"]

        title = work.get("display_name") or work.get("title") or "Unknown"
        year = str(work.get("publication_year", "N/A"))
        cite_count = work.get("cited_by_count", 0)
        abstract_inv = work.get("abstract_inverted_index")
        if abstract_inv and isinstance(abstract_inv, dict):
            word_pos = []
            for word, positions in abstract_inv.items():
                for pos in positions:
                    word_pos.append((pos, word))
            word_pos.sort()
            abstract = " ".join(w for _, w in word_pos)[:600]
        else:
            abstract = "No abstract available."

        text = (
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Year: {year}\n"
            f"Citations: {cite_count}\n"
            f"Abstract: {abstract}"
        )
        source_url = work.get("doi") or work.get("id", "")
        if source_url and source_url.startswith("https://doi.org/"):
            pass
        elif work.get("doi"):
            source_url = f"https://doi.org/{work['doi']}"

        passages.append({
            "text": text,
            "title": title,
            "source": source_url,
            "section": "openalex",
            "pdf_url": pdf_url,
            "year": year,
            "authors": authors,
            "citations": cite_count,
        })
    return passages


def execute_open_access_search(
    query: str,
    max_results: int = 5,
    year_range: str = "",
    open_access_only: bool = True,
) -> dict:
    # Try Semantic Scholar only when an API key is configured (avoids anonymous 429s)
    if _S2_API_KEY:
        params = {
            "query": query,
            "limit": min(max_results, 10),
            "fields": _S2_FIELDS,
        }
        if open_access_only:
            params["openAccessPdf"] = ""
        if year_range:
            params["year"] = year_range
        url = f"{_S2_API}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={"x-api-key": _S2_API_KEY, "User-Agent": "IndicRAG/2.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())
            passages = _parse_s2_papers(data)
            if passages:
                return {"passages": passages}
        except Exception as e:
            logger.warning(f"[OpenAccessSearch] Semantic Scholar failed: {e}")

    # Always fall through to OpenAlex (no key needed, no rate limit)
    logger.info("[OpenAccessSearch] Using OpenAlex API")
    try:
        passages = _fetch_openalex(query, max_results, year_range, open_access_only)
        if passages:
            return {"passages": passages}
    except Exception as e:
        logger.error(f"[OpenAccessSearch] OpenAlex failed: {e}")

    return {"passages": [{"text": "No open-access papers found.",
                          "source": "open_access_search",
                          "title": "No results", "section": "none"}]}


TOOL_DISPATCH = {
    "indicrag_retrieval": lambda args: execute_indicrag(
        args["query"], args.get("expand_query", False)
    ),
    "web_search": lambda args: execute_web_search(
        args["query"], args.get("num_results", 5)
    ),
    "calculate": lambda args: execute_calculate(args["expression"]),
    "execute_python": lambda args: execute_python(args["code"]),
    "arxiv_search": lambda args: execute_arxiv_search(
        args["query"],
        args.get("max_results", 5),
        args.get("sort_by", "relevance"),
        args.get("year_from"),
    ),
    "open_access_search": lambda args: execute_open_access_search(
        args["query"], args.get("max_results", 5),
        args.get("year_range", ""), args.get("open_access_only", True)
    ),
}
