"""
Zero-shot çeviri prompt'ları — EN↔TR çift yönlü.
"""

SYSTEM_PROMPT = (
    "You are a professional translator. "
    "Translate the given sentence accurately and naturally. "
    "Output only the translation, nothing else."
)


def build_zero_shot_prompt(src_text: str, src_lang: str, tgt_lang: str) -> tuple[str, str]:
    """
    (system, user) tuple döndürür; TranslationModel.generate() ile kullanılır.

    src_lang / tgt_lang: "English" veya "Turkish"
    """
    user = (
        f"Translate the following {src_lang} sentence into {tgt_lang}.\n\n"
        f"{src_lang}: {src_text}\n"
        f"{tgt_lang}:"
    )
    return SYSTEM_PROMPT, user


def detect_language_from_pair(pair: dict, direction: str) -> tuple[str, str, str]:
    """
    direction: "en2tr" | "tr2en"
    Döndürür: (src_text, src_lang_label, tgt_lang_label)
    """
    if direction == "en2tr":
        return pair["en"], "English", "Turkish"
    return pair["tr"], "Turkish", "English"
