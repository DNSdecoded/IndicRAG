"""
Phase 3 — multimodal figure/table indexing.

Extract figure/table regions and their captions from a PDF, have the Gemini VLM
describe each region, and return them as retrievable chunks whose text is
``caption + VLM description``. Chunks are indexed into the *same* vector store as
text chunks (see ingest.py), so retrieval and citation need no changes.

Two stages, to fit the existing ingest pipeline:
  1. ``extract_regions`` — CPU only (PyMuPDF). Safe inside the ingest ProcessPool
     worker. Returns picklable dicts carrying PNG bytes + nearest caption.
  2. ``caption_regions`` — the network VLM call. Parent-side only, after dedup,
     so duplicate/unchanged papers are never captioned (mirrors how arXiv
     enrichment is kept out of the workers). Crops are written to disk *here*,
     only for regions that survive into an indexed chunk — no orphan PNGs.

Gated by ``config.ENABLE_MULTIMODAL_INGEST`` in the caller. Per-doc cost bounded
by ``config.MULTIMODAL_MAX_FIGS_PER_DOC``.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF (already a project dependency via pdf_utils)

import config

logger = logging.getLogger(__name__)

# A text block is a caption if it opens with "Figure 3", "Fig. 2", "Table 1",
# "Tab 4" — case-insensitive, tolerant of the label/number separator.
_CAPTION_RE = re.compile(r"^\s*(fig(?:ure)?|tab(?:le)?)\b[\s.:]*\d", re.IGNORECASE)

_CROP_DPI = 150       # enough for the VLM to read axis labels without huge PNGs
_MIN_SIDE = 40        # skip logos / rules / hairlines
_CAPTION_MAX = 500    # trim runaway caption blocks
_TABLE_MD_MAX = 1500


def _kind_of(caption: str) -> str:
    """'table' if the caption starts with Table/Tab, else 'figure'."""
    return "table" if re.match(r"\s*tab", caption, re.IGNORECASE) else "figure"


def _safe_id(paper_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", paper_id)[:120]


def _caption_prompt(kind: str, caption: str) -> str:
    label = "table" if kind == "table" else "figure"
    hint = f' Its caption reads: "{caption}".' if caption else ""
    return (
        f"This is a {label} from a scientific paper.{hint} "
        f"Describe it densely for search retrieval: what it shows, the axes or "
        f"columns, key values, trends, and any stated result. Be concrete and "
        f"factual. One short paragraph, no preamble."
    )


def _nearest_caption(page: "fitz.Page", rect: "fitz.Rect") -> str:
    """Nearest labelled caption block (vertically) to the region, or ''."""
    best, best_dist = "", 1e9
    for block in page.get_text("blocks"):
        # block = (x0, y0, x1, y1, text, block_no, block_type)
        text = (block[4] or "").strip()
        if not _CAPTION_RE.match(text):
            continue
        by0 = block[1]
        dist = min(abs(by0 - rect.y1), abs(by0 - rect.y0))
        if dist < best_dist:
            best, best_dist = " ".join(text.split()), dist
    return best[:_CAPTION_MAX]


def extract_regions(
    pdf_path: str, paper_id: str, max_figs: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Extract up to ``max_figs`` figure/table regions from a PDF (CPU only).

    Sources: raster images (``page.get_images``) and detected tables
    (``page.find_tables``). Renders each region to PNG bytes in memory — nothing
    is written to disk here; ``caption_regions`` persists only the kept crops.

    Returns picklable dicts: {kind, page, idx, caption, table_md, png}.
    """
    if max_figs is None:
        max_figs = config.MULTIMODAL_MAX_FIGS_PER_DOC

    regions: List[Dict[str, Any]] = []
    try:
        with fitz.open(pdf_path) as doc:
            for pno in range(len(doc)):
                if len(regions) >= max_figs:
                    break
                page = doc[pno]
                rects: List[tuple] = []  # (rect, table_md)

                # Raster figures: placement rect of each embedded image.
                for img in page.get_images(full=True):
                    for r in page.get_image_rects(img[0]):
                        if r.width > _MIN_SIDE and r.height > _MIN_SIDE:
                            rects.append((r, ""))

                # Tables (vector-drawn, invisible to get_images).
                try:
                    for tbl in page.find_tables().tables:
                        rects.append((fitz.Rect(tbl.bbox), tbl.to_markdown()))
                except Exception as tbl_err:  # find_tables is best-effort
                    logger.debug("find_tables failed on page %d: %s", pno, tbl_err)

                for idx, (rect, table_md) in enumerate(rects):
                    if len(regions) >= max_figs:
                        break
                    caption = _nearest_caption(page, rect)
                    kind = "table" if table_md else _kind_of(caption)
                    try:
                        png = page.get_pixmap(clip=rect, dpi=_CROP_DPI).tobytes("png")
                    except Exception as crop_err:
                        logger.debug("crop render failed p%d #%d: %s", pno, idx, crop_err)
                        continue
                    regions.append({
                        "kind": kind,
                        "page": pno + 1,
                        "idx": idx,
                        "caption": caption,
                        "table_md": table_md[:_TABLE_MD_MAX],
                        "png": png,
                    })
    except Exception as e:
        logger.warning("Figure extraction failed for %s: %s", pdf_path, e)
    return regions


def _caption_one(region: Dict[str, Any]) -> str:
    """VLM-describe a single region. Returns '' on failure (best-effort)."""
    from google.genai import types
    import llm_client
    import rag

    gen_config = types.GenerateContentConfig(
        system_instruction=(
            "You are a scientific figure and table analyst: describe what the image "
            "concretely shows, factually and without preamble."
        ),
        temperature=0.0,
        max_output_tokens=512,
        safety_settings=config.SAFETY_SETTINGS,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    contents = [
        types.Part.from_bytes(data=region["png"], mime_type="image/png"),
        _caption_prompt(region["kind"], region["caption"]),
    ]
    try:
        resp = llm_client.generate_with_failover(config.LLM_MODEL_NAME, contents, gen_config)
        return rag.safe_extract_text(resp).strip()
    except Exception as e:
        logger.warning("VLM caption failed (page %s): %s", region.get("page"), e)
        return ""


def caption_regions(
    regions: List[Dict[str, Any]], paper_id: str
) -> List[Dict[str, Any]]:
    """Caption each region (network) and build figure chunk dicts.

    Parent-side only. Captioning fans out across a small thread pool — a
    figure-heavy paper used to serialize a dozen VLM round-trips into ingest
    latency. Concurrency is deliberately small and bounded
    (`config.FIGURE_CAPTION_WORKERS`): unbounded fan-out just trades a slow
    ingest for a 429 storm that trips the LLM circuit breaker for everything
    else running at the time.

    A region's crop is written to disk only if it yields an indexable chunk, so
    dropped regions leave no orphan PNGs.

    Returns chunk dicts: {text, chunk_type, page, crop_path, caption}, in region
    order regardless of which caption finished first. Regions the VLM can't
    describe and that also carry no caption/table text are dropped.
    """
    out_dir = config.FIGURES_DIR / _safe_id(paper_id)
    chunks: List[Dict[str, Any]] = []

    workers = max(1, min(config.FIGURE_CAPTION_WORKERS, len(regions))) if regions else 1
    if workers == 1:
        descs = [_caption_one(r) for r in regions]
    else:
        # map() yields results in submission order, so chunk order still follows
        # page/figure order and does not depend on completion order.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vlm-caption") as pool:
            descs = list(pool.map(_caption_one, regions))

    for r, desc in zip(regions, descs):
        parts = [p for p in (r["caption"], desc, r["table_md"]) if p]
        body = "\n".join(parts).strip()
        if not body:
            continue  # no caption, no table, no VLM output — nothing to index

        out_dir.mkdir(parents=True, exist_ok=True)
        crop_path = out_dir / f"{r['page']}_{r['idx']}.png"
        crop_written = True
        try:
            crop_path.write_bytes(r["png"])
        except Exception as e:  # a missing crop must not lose the text chunk
            logger.warning("crop write failed %s: %s", crop_path, e)
            crop_written = False

        label = "Table" if r["kind"] == "table" else "Figure"
        chunks.append({
            "text": f"[{label} on page {r['page']}] {body}",
            "chunk_type": r["kind"],
            "page": r["page"],
            # None when the write failed — don't record a path to a nonexistent
            # file, or the UI would render a broken <img>/404.
            "crop_path": str(crop_path) if crop_written else None,
            "caption": r["caption"],
        })
    return chunks


if __name__ == "__main__":
    # Self-check: chunk assembly is correct without hitting the VLM or a real PDF.
    import pathlib
    import tempfile

    logging.basicConfig(level=logging.INFO)

    assert _kind_of("Table 1: results") == "table"
    assert _kind_of("Figure 2. antenna") == "figure"
    assert _kind_of("no label here") == "figure"
    assert "table" in _caption_prompt("table", "Table 1").lower()

    assert _CAPTION_RE.match("Figure 3: S11 vs frequency")
    assert _CAPTION_RE.match("Tab. 2 comparison")
    assert not _CAPTION_RE.match("The figure shows a decline")

    import figure_captioner as fc

    with tempfile.TemporaryDirectory() as tmp:
        config.FIGURES_DIR = pathlib.Path(tmp)  # write crops to a temp dir
        fc._caption_one = lambda r: "resonance dip at 2.4 GHz"  # type: ignore
        regions = [
            {"kind": "figure", "page": 5, "idx": 0, "caption": "Figure 3: S11",
             "table_md": "", "png": b"\x89PNG"},
            {"kind": "table", "page": 6, "idx": 1, "caption": "",
             "table_md": "| a | b |", "png": b"\x89PNG"},
        ]
        chunks = fc.caption_regions(regions, "paper-1")
        assert len(chunks) == 2, chunks
        assert chunks[0]["text"].startswith("[Figure on page 5]")
        assert "resonance dip" in chunks[0]["text"]
        assert chunks[0]["chunk_type"] == "figure"
        assert chunks[1]["text"].startswith("[Table on page 6]")
        assert chunks[0]["crop_path"].endswith("5_0.png")

        # Region with no caption, no table, empty VLM output → dropped, no crop.
        fc._caption_one = lambda r: ""  # type: ignore
        empty = fc.caption_regions(
            [{"kind": "figure", "page": 1, "idx": 0, "caption": "",
              "table_md": "", "png": b"\x89PNG"}],
            "paper-2",
        )
        assert empty == [], empty
        assert not (config.FIGURES_DIR / "paper-2").exists(), "dropped region wrote a crop"

    print("figure_captioner self-check passed")
