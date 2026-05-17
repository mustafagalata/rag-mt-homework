"""
MAPS (Multi-Aspect Prompting and Selection) pipeline.

He et al., TACL 2024 — "Exploring Human-Like Translation Strategy with LLMs"
DOI: 10.1162/tacl_a_00642

3 aşama:
    1. Knowledge Mining  — keywords, topics, demonstrations çıkar
    2. Knowledge Integration — her bilgi tipi ile aday çeviri üret (+ baseline)
    3. Knowledge Selection — LLM-as-Judge (SCQ) ile en iyi adayı seç
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 5-shot exemplars (makale dipnot 1: her mining adımı için manuel hazırlanmış)
# EN→TR ve TR→EN için ayrı exemplar setleri
# ──────────────────────────────────────────────────────────────────────────────

KEYWORD_EXEMPLARS_EN2TR = """Example 1:
Sentence: The government announced new economic policies to combat inflation.
Keywords: government→hükümet | economic policies→ekonomi politikaları | combat→mücadele etmek | inflation→enflasyon

Example 2:
Sentence: Scientists discovered a new species of deep-sea fish near the Pacific Ocean.
Keywords: scientists→bilim insanları | discovered→keşfetti | species→tür | deep-sea→derin deniz | Pacific Ocean→Pasifik Okyanusu

Example 3:
Sentence: The hospital introduced a new treatment method for cancer patients.
Keywords: hospital→hastane | treatment method→tedavi yöntemi | cancer→kanser | patients→hastalar

Example 4:
Sentence: The football match ended with a dramatic penalty shootout.
Keywords: football match→futbol maçı | dramatic→dramatik | penalty shootout→penaltı atışları

Example 5:
Sentence: Renewable energy sources are becoming increasingly important for sustainable development.
Keywords: renewable energy→yenilenebilir enerji | sustainable development→sürdürülebilir kalkınma | increasingly→giderek"""

KEYWORD_EXEMPLARS_TR2EN = """Example 1:
Sentence: Hükümet enflasyonla mücadele etmek için yeni ekonomi politikaları açıkladı.
Keywords: hükümet→government | ekonomi politikaları→economic policies | mücadele etmek→combat | enflasyon→inflation

Example 2:
Sentence: Bilim insanları Pasifik Okyanusu yakınlarında yeni bir derin deniz balığı türü keşfetti.
Keywords: bilim insanları→scientists | keşfetti→discovered | tür→species | derin deniz→deep-sea | Pasifik Okyanusu→Pacific Ocean

Example 3:
Sentence: Hastane kanser hastaları için yeni bir tedavi yöntemi tanıttı.
Keywords: hastane→hospital | tedavi yöntemi→treatment method | kanser→cancer | hastalar→patients

Example 4:
Sentence: Futbol maçı dramatik bir penaltı atışıyla sona erdi.
Keywords: futbol maçı→football match | dramatik→dramatic | penaltı atışları→penalty shootout

Example 5:
Sentence: Yenilenebilir enerji kaynakları sürdürülebilir kalkınma için giderek daha önemli hale geliyor.
Keywords: yenilenebilir enerji→renewable energy | sürdürülebilir kalkınma→sustainable development | giderek→increasingly"""

TOPIC_EXEMPLARS_EN2TR = """Example 1:
Sentence: The government announced new economic policies to combat inflation.
Topic: Politics and Economics — government economic policy and inflation management

Example 2:
Sentence: Scientists discovered a new species of deep-sea fish near the Pacific Ocean.
Topic: Science and Nature — marine biology discovery

Example 3:
Sentence: The hospital introduced a new treatment method for cancer patients.
Topic: Medicine and Health — oncology treatment innovation

Example 4:
Sentence: The football match ended with a dramatic penalty shootout.
Topic: Sports — football match result

Example 5:
Sentence: Renewable energy sources are becoming increasingly important for sustainable development.
Topic: Environment and Energy — renewable energy and sustainability"""

TOPIC_EXEMPLARS_TR2EN = """Example 1:
Sentence: Hükümet enflasyonla mücadele etmek için yeni ekonomi politikaları açıkladı.
Topic: Politics and Economics — government economic policy and inflation management

Example 2:
Sentence: Bilim insanları Pasifik Okyanusu yakınlarında yeni bir derin deniz balığı türü keşfetti.
Topic: Science and Nature — marine biology discovery

Example 3:
Sentence: Hastane kanser hastaları için yeni bir tedavi yöntemi tanıttı.
Topic: Medicine and Health — oncology treatment innovation

Example 4:
Sentence: Futbol maçı dramatik bir penaltı atışıyla sona erdi.
Topic: Sports — football match result

Example 5:
Sentence: Yenilenebilir enerji kaynakları sürdürülebilir kalkınma için giderek daha önemli hale geliyor.
Topic: Environment and Energy — renewable energy and sustainability"""

DEMO_EXEMPLARS_EN2TR = """Example 1:
Sentence: The government announced new economic policies to combat inflation.
Similar translation pairs:
- EN: The authorities unveiled a fresh fiscal strategy. | TR: Yetkililer yeni bir maliye stratejisi açıkladı.
- EN: Officials presented measures to control rising prices. | TR: Yetkililer artan fiyatları kontrol etmek için önlemler sundu.
- EN: The administration revealed plans to stabilize the economy. | TR: Yönetim ekonomiyi istikrara kavuşturma planlarını açıkladı.

Example 2:
Sentence: Scientists discovered a new species of deep-sea fish.
Similar translation pairs:
- EN: Researchers identified an unknown marine organism. | TR: Araştırmacılar bilinmeyen bir deniz organizması tespit etti.
- EN: A team of biologists found a rare underwater creature. | TR: Bir biyolog ekibi nadir bir su altı canlısı buldu.
- EN: The expedition revealed a previously unseen aquatic species. | TR: Keşif gezisi daha önce görülmemiş bir su canlısı türünü ortaya çıkardı."""

DEMO_EXEMPLARS_TR2EN = """Example 1:
Sentence: Hükümet enflasyonla mücadele etmek için yeni ekonomi politikaları açıkladı.
Similar translation pairs:
- TR: Yetkililer yeni bir maliye stratejisi açıkladı. | EN: The authorities unveiled a fresh fiscal strategy.
- TR: Yetkililer artan fiyatları kontrol etmek için önlemler sundu. | EN: Officials presented measures to control rising prices.
- TR: Yönetim ekonomiyi istikrara kavuşturma planlarını açıkladı. | EN: The administration revealed plans to stabilize the economy.

Example 2:
Sentence: Bilim insanları yeni bir derin deniz balığı türü keşfetti.
Similar translation pairs:
- TR: Araştırmacılar bilinmeyen bir deniz organizması tespit etti. | EN: Researchers identified an unknown marine organism.
- TR: Bir biyolog ekibi nadir bir su altı canlısı buldu. | EN: A team of biologists found a rare underwater creature.
- TR: Keşif gezisi daha önce görülmemiş bir su canlısı türünü ortaya çıkardı. | EN: The expedition revealed a previously unseen aquatic species."""


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builder fonksiyonları
# ──────────────────────────────────────────────────────────────────────────────

MAPS_SYSTEM = (
    "You are a professional translator and linguistic analyst. "
    "Follow the instructions precisely and output only what is asked."
)


def _keyword_prompt(src: str, src_lang: str, tgt_lang: str) -> str:
    exemplars = KEYWORD_EXEMPLARS_EN2TR if src_lang == "English" else KEYWORD_EXEMPLARS_TR2EN
    return (
        f"Extract the important keywords from the {src_lang} sentence and provide "
        f"their {tgt_lang} translations in the format: word→translation | word→translation\n\n"
        f"{exemplars}\n\n"
        f"Now extract keywords:\n"
        f"Sentence: {src}\n"
        f"Keywords:"
    )


def _topic_prompt(src: str, src_lang: str) -> str:
    exemplars = TOPIC_EXEMPLARS_EN2TR if src_lang == "English" else TOPIC_EXEMPLARS_TR2EN
    return (
        f"Identify the topic of the following {src_lang} sentence. "
        f"Be concise and specific (one line).\n\n"
        f"{exemplars}\n\n"
        f"Now identify the topic:\n"
        f"Sentence: {src}\n"
        f"Topic:"
    )


def _demo_prompt(src: str, src_lang: str, tgt_lang: str) -> str:
    exemplars = DEMO_EXEMPLARS_EN2TR if src_lang == "English" else DEMO_EXEMPLARS_TR2EN
    return (
        f"Generate 3 similar {src_lang}→{tgt_lang} translation pairs "
        f"that could help translate the given sentence.\n\n"
        f"{exemplars}\n\n"
        f"Now generate similar pairs for:\n"
        f"Sentence: {src}\n"
        f"Similar translation pairs:"
    )


def _translate_with_knowledge(
    src: str,
    src_lang: str,
    tgt_lang: str,
    knowledge_type: str,    # "keywords" | "topic" | "demonstrations" | "none"
    knowledge_text: str = "",
) -> str:
    """Integration adımı için prompt oluşturur."""
    base = (
        f"Translate the following {src_lang} sentence into {tgt_lang}. "
        f"Output only the translation.\n\n"
    )
    if knowledge_type == "keywords" and knowledge_text:
        guidance = f"Relevant keywords: {knowledge_text}\n\n"
    elif knowledge_type == "topic" and knowledge_text:
        guidance = f"Topic context: {knowledge_text}\n\n"
    elif knowledge_type == "demonstrations" and knowledge_text:
        guidance = f"Similar translation examples:\n{knowledge_text}\n\n"
    else:
        guidance = ""

    return (
        base
        + guidance
        + f"{src_lang}: {src}\n"
        + f"{tgt_lang}:"
    )


def _selection_prompt(
    src: str,
    src_lang: str,
    tgt_lang: str,
    candidates: list[str],
) -> str:
    """LLM-SCQ (Single Choice Question) — LLM-as-Judge."""
    options = "\n".join(
        f"{chr(65 + i)}) {c}" for i, c in enumerate(candidates)
    )
    return (
        f"Which of the following {tgt_lang} translations of the {src_lang} sentence "
        f"is the most accurate and natural?\n\n"
        f"{src_lang}: {src}\n\n"
        f"Translations:\n{options}\n\n"
        f"Answer with the letter only (A, B, C, or D):"
    )


def _parse_selection(raw: str, n_candidates: int) -> int:
    """Modelin cevabından seçilen index'i çıkar (0-indexed). Fallback: 0."""
    raw = raw.strip().upper()
    for i in range(n_candidates):
        if chr(65 + i) in raw:
            return i
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Ana MAPS sınıfı
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MAPSResult:
    src: str
    src_lang: str
    tgt_lang: str
    keywords: str = ""
    topic: str = ""
    demonstrations: str = ""
    candidate_baseline: str = ""
    candidate_keywords: str = ""
    candidate_topic: str = ""
    candidate_demo: str = ""
    selected_index: int = 0
    final_translation: str = ""
    candidates: list[str] = field(default_factory=list)


class MAPSPipeline:

    def __init__(self, translation_model, config: dict):
        self.model = translation_model
        self.max_mining = config["model"]["max_new_tokens_mining"]
        self.max_trans = config["model"]["max_new_tokens_translation"]
        self.max_sel = config["model"]["max_new_tokens_selection"]

    def run(self, src: str, src_lang: str, tgt_lang: str) -> MAPSResult:
        result = MAPSResult(src=src, src_lang=src_lang, tgt_lang=tgt_lang)

        # ── Aşama 1: Knowledge Mining ──────────────────────────────────────
        result.keywords = self.model.generate(
            MAPS_SYSTEM,
            _keyword_prompt(src, src_lang, tgt_lang),
            max_new_tokens=self.max_mining,
        )
        result.topic = self.model.generate(
            MAPS_SYSTEM,
            _topic_prompt(src, src_lang),
            max_new_tokens=self.max_mining,
        )
        result.demonstrations = self.model.generate(
            MAPS_SYSTEM,
            _demo_prompt(src, src_lang, tgt_lang),
            max_new_tokens=self.max_mining,
        )

        # ── Aşama 2: Knowledge Integration (4 aday) ───────────────────────
        result.candidate_baseline = self.model.generate(
            MAPS_SYSTEM,
            _translate_with_knowledge(src, src_lang, tgt_lang, "none"),
            max_new_tokens=self.max_trans,
        )
        result.candidate_keywords = self.model.generate(
            MAPS_SYSTEM,
            _translate_with_knowledge(src, src_lang, tgt_lang, "keywords", result.keywords),
            max_new_tokens=self.max_trans,
        )
        result.candidate_topic = self.model.generate(
            MAPS_SYSTEM,
            _translate_with_knowledge(src, src_lang, tgt_lang, "topic", result.topic),
            max_new_tokens=self.max_trans,
        )
        result.candidate_demo = self.model.generate(
            MAPS_SYSTEM,
            _translate_with_knowledge(src, src_lang, tgt_lang, "demonstrations", result.demonstrations),
            max_new_tokens=self.max_trans,
        )

        result.candidates = [
            result.candidate_baseline,
            result.candidate_keywords,
            result.candidate_topic,
            result.candidate_demo,
        ]

        # ── Aşama 3: Knowledge Selection (LLM-as-Judge) ───────────────────
        sel_prompt = _selection_prompt(src, src_lang, tgt_lang, result.candidates)
        sel_raw = self.model.generate(
            MAPS_SYSTEM,
            sel_prompt,
            max_new_tokens=self.max_sel,
        )
        result.selected_index = _parse_selection(sel_raw, len(result.candidates))
        result.final_translation = result.candidates[result.selected_index]

        return result
