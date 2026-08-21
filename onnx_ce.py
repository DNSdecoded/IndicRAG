"""int8 ONNX loader for CPU cross-encoders (reranker + NLI faithfulness).

On this CPU-only box the fp32 cross-encoders run at ~10s/pair (NLI) and ~35s/15-pairs
(reranker); int8 dynamic-quantized ONNX is ~11x / ~3x faster with matching scores.

Self-bootstrapping: the first load builds the quantized model from the local HF
snapshot (works fully offline — export runs on the cached torch weights) and caches it
under models/onnx_ce/<subdir>/. Callers wrap load() in try/except and fall back to the
fp32 torch CrossEncoder, so a box without optimum/onnx never hard-fails.
"""
import glob
import logging
import os
import shutil
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_QUANT = "avx2"  # widely-supported int8 preset; runs on any x86-64
_QFILE = f"onnx/model_quint8_{_QUANT}.onnx"


def _snapshot_dir(model_id: str) -> str | None:
    """Newest COMPLETE local HF snapshot dir for a repo id, or None if not cached.

    Completeness matters, not just recency. A repo can have several snapshots —
    a PR revision often lands as a partial one holding just model.safetensors,
    with no config.json or tokenizer. Picking purely by mtime selects that
    partial dir, and the export then fails deep inside transformers with an
    unrelated-looking error (`A configuration of type rag cannot be
    instantiated...`), because AutoConfig cannot find a config to read.

    config.json is the marker: without it nothing downstream can load the model.
    """
    safe = "models--" + model_id.replace("/", "--")
    snaps = glob.glob(str(config.MODELS_CACHE_DIR / safe / "snapshots" / "*"))
    complete = [s for s in snaps if os.path.isfile(os.path.join(s, "config.json"))]
    if not complete:
        return None
    return max(complete, key=os.path.getmtime)


def _model_class(kind: str):
    """CrossEncoder for rerank/NLI, SentenceTransformer for the bi-encoder embedder.

    Both quantize through the same export path; only the wrapper class differs.
    """
    from sentence_transformers import CrossEncoder, SentenceTransformer
    return SentenceTransformer if kind == "bi" else CrossEncoder


def _build(model_id: str, out: Path, kind: str = "cross") -> None:
    """Export int8 ONNX from the local snapshot and drop config/tokenizer beside it."""
    from sentence_transformers.backend import export_dynamic_quantized_onnx_model

    snap = _snapshot_dir(model_id)
    if not snap:
        raise FileNotFoundError(f"No local snapshot for {model_id}; pre-download it first")

    # backend="onnx" uses the snapshot's onnx/ if present, else auto-exports fp32 from
    # the torch weights (offline OK). export_dynamic_quantized_onnx_model requires an
    # onnx-backend model, then writes onnx/model_quint8_<quant>.onnx into `out`.
    base = _model_class(kind)(snap, backend="onnx", local_files_only=True)
    out.mkdir(parents=True, exist_ok=True)
    export_dynamic_quantized_onnx_model(base, _QUANT, str(out))

    # The quantized .onnx alone isn't loadable — copy the config + tokenizer (not the
    # big fp32 weights) so `out` is a self-contained, offline-loadable model dir.
    #
    # Subdirectories matter for bi-encoders: modules.json points at 1_Pooling/ (and
    # sometimes 2_Normalize/), and without them SentenceTransformer builds Pooling
    # with no config and fails with `missing 1 required positional argument:
    # 'embedding_dimension'`. Cross-encoders have no such dirs, so this is a no-op
    # for them.
    #
    # Every fp32 weight format is excluded, not just .safetensors: bge-m3 ships
    # pytorch_model.bin, and copying it added 2.3 GB of weights that the ONNX
    # runtime never reads.
    weight_suffixes = (".safetensors", ".bin", ".h5", ".msgpack", ".ckpt", ".pth", ".pt")
    for name in os.listdir(snap):
        src = os.path.join(snap, name)
        if os.path.isfile(src):
            if not name.endswith(weight_suffixes):
                shutil.copy2(src, out / name)
        elif os.path.isdir(src) and name != "onnx":
            # dirs_exist_ok: a rebuild over an existing export must not fail.
            shutil.copytree(src, out / name, dirs_exist_ok=True)


def load(model_id: str, subdir: str, kind: str = "cross"):
    """Return an int8-ONNX model for model_id, building it once if needed.

    kind="cross" -> CrossEncoder (reranker, NLI). kind="bi" -> SentenceTransformer
    (the embedding model), where the win matters most: embedding is the dominant
    cost of a bulk ingest, and it ran fp32 while the cross-encoders were already
    quantized.

    Raises on any failure — the caller decides whether to fall back to fp32.
    """
    out = config.MODELS_CACHE_DIR / "onnx_ce" / subdir
    if not (out / _QFILE).exists():
        logger.info(f"[onnx_ce] building int8 ONNX for {model_id} (one-time, ~90s)...")
        _build(model_id, out, kind)
        logger.info(f"[onnx_ce] built {out}")
    return _model_class(kind)(str(out), backend="onnx",
                              model_kwargs={"file_name": _QFILE}, local_files_only=True)
