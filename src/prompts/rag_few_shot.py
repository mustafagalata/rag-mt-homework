"""
RAG tabanlı 5-shot prompt oluşturucu.
Retrieved örnekler in-context demonstration olarak prompt'a eklenir.
"""

SYSTEM_PROMPT = (
    "You are a professional translator. "
    "Use the provided examples as guidance to produce an accurate and natural translation. "
    "Output only the translation, nothing else."
)


def build_rag_prompt(
    src_text: str,
    src_lang: str,
    tgt_lang: str,
    retrieved_pairs: list[dict],
) -> tuple[str, str]:
    """
    (system, user) tuple döndürür.

    retrieved_pairs: Retriever'dan gelen [{"en": ..., "tr": ...}, ...] listesi.
    Kaynak dile göre doğru tarafı src/tgt olarak ayarlar.
    """
    src_key = "en" if src_lang == "English" else "tr"
    tgt_key = "tr" if src_lang == "English" else "en"

    examples = "\n".join(
        f"Example {i + 1}:\n"
        f"  {src_lang}: {p[src_key]}\n"
        f"  {tgt_lang}: {p[tgt_key]}"
        for i, p in enumerate(retrieved_pairs)
    )

    user = (
        f"Here are {len(retrieved_pairs)} similar translation examples:\n\n"
        f"{examples}\n\n"
        f"Now translate the following sentence:\n"
        f"{src_lang}: {src_text}\n"
        f"{tgt_lang}:"
    )
    return SYSTEM_PROMPT, user
