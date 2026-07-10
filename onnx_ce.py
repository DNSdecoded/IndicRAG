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
    """Newest local HF snapshot dir for a repo id, or None if not cached."""
    safe = "models--" + model_id.replace("/", "--")
    snaps = glob.glob(str(config.MODELS_CACHE_DIR / safe / "snapshots" / "*"))
    return max(snaps, key=os.path.getmtime) if snaps else None


def _build(model_id: str, out: Path) -> None:
    """Export int8 ONNX from the local snapshot and drop config/tokenizer beside it."""
    from sentence_transformers import CrossEncoder
    from sentence_transformers.backend import export_dynamic_quantized_onnx_model

    snap = _snapshot_dir(model_id)
    if not snap:
        raise FileNotFoundError(f"No local snapshot for {model_id}; pre-download it first")

    # backend="onnx" uses the snapshot's onnx/ if present, else auto-exports fp32 from
    # the torch weights (offline OK). export_dynamic_quantized_onnx_model requires an
    # onnx-backend model, then writes onnx/model_quint8_<quant>.onnx into `out`.
    base = CrossEncoder(snap, backend="onnx", local_files_only=True)
    out.mkdir(parents=True, exist_ok=True)
    export_dynamic_quantized_onnx_model(base, _QUANT, str(out))

    # The quantized .onnx alone isn't loadable — copy the config + tokenizer (not the
    # big fp32 weights) so `out` is a self-contained, offline-loadable model dir.
    for name in os.listdir(snap):
        src = os.path.join(snap, name)
        if os.path.isfile(src) and not name.endswith(".safetensors"):
            shutil.copy2(src, out / name)


def load(model_id: str, subdir: str):
    """Return an int8-ONNX CrossEncoder for model_id, building it once if needed.

    Raises on any failure — the caller decides whether to fall back to fp32.
    """
    from sentence_transformers import CrossEncoder

    out = config.MODELS_CACHE_DIR / "onnx_ce" / subdir
    if not (out / _QFILE).exists():
        logger.info(f"[onnx_ce] building int8 ONNX for {model_id} (one-time, ~90s)...")
        _build(model_id, out)
        logger.info(f"[onnx_ce] built {out}")
    return CrossEncoder(str(out), backend="onnx",
                        model_kwargs={"file_name": _QFILE}, local_files_only=True)
