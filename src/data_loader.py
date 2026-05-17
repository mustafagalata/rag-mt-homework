"""
WMT16 EN-TR veri seti: yükleme, ön işleme ve subset oluşturma.
"""

import random
import logging
from pathlib import Path
from typing import Optional

import yaml
import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_valid_pair(en: str, tr: str, min_tok: int, max_tok: int) -> bool:
    en_words = en.split()
    tr_words = tr.split()
    if not (min_tok <= len(en_words) <= max_tok):
        return False
    if not (min_tok <= len(tr_words) <= max_tok):
        return False
    # Kaynak ve hedef birebir aynıysa at
    if en.strip().lower() == tr.strip().lower():
        return False
    return True


def preprocess_split(raw_split, min_tokens: int, max_tokens: int) -> list[dict]:
    """
    Ham HuggingFace split'inden geçerli (en, tr) çiftlerini döndürür.
    WMT16 tr-en formatı: item["translation"] = {"en": ..., "tr": ...}
    """
    pairs = []
    for item in raw_split:
        en = item["translation"]["en"].strip()
        tr = item["translation"]["tr"].strip()
        if _is_valid_pair(en, tr, min_tokens, max_tokens):
            pairs.append({"en": en, "tr": tr})
    return pairs


def create_subset(pairs: list[dict], n: int | None, seed: int) -> list[dict]:
    """n=None ise listenin tamamını döndürür (full set)."""
    if n is None or n >= len(pairs):
        return list(pairs)
    rng = random.Random(seed)
    return rng.sample(pairs, n)


def load_wmt16_splits(config: dict, cache_dir: Optional[str] = None) -> dict:
    """
    Döndürür:
        train      : RAG corpus için 50 000 çift (train'den)
        test_en2tr : değerlendirme için 500 çift, EN kaynak
        test_tr2en : değerlendirme için 500 çift, TR kaynak
                     (iki yön arasında çakışma yok)
    """
    logger.info("WMT16 tr-en yükleniyor...")
    ds = load_dataset(
        "wmt16",
        "tr-en",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    cfg = config["data"]
    seed = cfg["seed"]
    min_tok = cfg["min_src_tokens"]
    max_tok = cfg["max_src_tokens"]

    # ── Train ──────────────────────────────────────────────────────────────
    logger.info("Train seti işleniyor...")
    train_pairs = preprocess_split(ds["train"], min_tok, max_tok)
    train_subset = create_subset(train_pairs, cfg["train_subset_size"], seed)
    logger.info(f"Train subset: {len(train_subset):,} çift")

    # ── Test ───────────────────────────────────────────────────────────────
    logger.info("Test seti işleniyor...")
    test_pairs = preprocess_split(ds["test"], min_tok, max_tok)
    logger.info(f"Test (temiz toplam): {len(test_pairs):,} çift")

    # İki yön için ÇAKIŞMAYAN subset — seed'li shuffle sonra dilimleme
    n = cfg["test_subset_size"]
    rng = random.Random(seed)
    shuffled = list(test_pairs)
    rng.shuffle(shuffled)

    test_en2tr = shuffled[:n]           # EN → TR değerlendirme seti
    test_tr2en = shuffled[n: n * 2]     # TR → EN değerlendirme seti

    if len(test_tr2en) < n:
        logger.warning(
            f"Test seti yeterince büyük değil: "
            f"TR→EN için yalnızca {len(test_tr2en)} cümle alınabildi."
        )

    logger.info(f"Test EN→TR: {len(test_en2tr)} | TR→EN: {len(test_tr2en)}")

    return {
        "train": train_subset,
        "test_en2tr": test_en2tr,
        "test_tr2en": test_tr2en,
    }


def save_splits(splits: dict, data_dir: str) -> None:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    for name, pairs in splits.items():
        path = Path(data_dir) / f"{name}.jsonl"
        pd.DataFrame(pairs).to_json(
            path, orient="records", lines=True, force_ascii=False
        )
        logger.info(f"Kaydedildi: {path} ({len(pairs):,} kayıt)")


def load_splits_from_disk(data_dir: str) -> dict:
    result = {}
    for fpath in sorted(Path(data_dir).glob("*.jsonl")):
        result[fpath.stem] = pd.read_json(
            fpath, orient="records", lines=True
        ).to_dict("records")
    return result


def print_sample_pairs(splits: dict, n: int = 5) -> None:
    """Rapora konulacak örnek çiftleri yazdırır."""
    pairs = splits.get("test_en2tr", [])[:n]
    print("=== Örnek Çeviri Çiftleri (EN → TR) ===")
    for i, p in enumerate(pairs, 1):
        print(f"\n[{i}] EN: {p['en']}")
        print(f"     TR: {p['tr']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config()
    splits = load_wmt16_splits(cfg, cache_dir=cfg["paths"]["cache_dir"])
    save_splits(splits, cfg["paths"]["data_dir"])
    print_sample_pairs(splits)
