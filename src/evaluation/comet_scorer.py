"""
COMET skorlama wrapper (unbabel-comet).
wmt22-comet-da: kaynak + aday + referans ile skorlama.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def load_comet_model(model_name: str = "Unbabel/wmt22-comet-da", cache_dir: Optional[str] = None):
    from comet import download_model, load_from_checkpoint
    logger.info(f"COMET modeli yükleniyor: {model_name}")
    checkpoint = download_model(model_name, saving_directory=cache_dir)
    return load_from_checkpoint(checkpoint)


def score(
    comet_model,
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    batch_size: int = 8,
) -> dict:
    """
    Döndürür:
        {
            "scores": [float, ...],   # cümle düzeyinde
            "system_score": float,    # korpus ortalaması
        }
    """
    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]
    output = comet_model.predict(data, batch_size=batch_size, progress_bar=True)
    return {
        "scores": output.scores,
        "system_score": output.system_score,
    }


def score_direction(
    comet_model,
    pairs: list[dict],
    hypotheses: list[str],
    direction: str,     # "en2tr" | "tr2en"
    batch_size: int = 8,
) -> dict:
    """
    pairs: [{"en": ..., "tr": ...}, ...]
    direction: kaynak ve referans dilini belirler.
    """
    if direction == "en2tr":
        sources = [p["en"] for p in pairs]
        references = [p["tr"] for p in pairs]
    else:
        sources = [p["tr"] for p in pairs]
        references = [p["en"] for p in pairs]

    return score(comet_model, sources, hypotheses, references, batch_size)
