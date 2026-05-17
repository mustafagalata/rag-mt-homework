"""
Qwen 2.5 7B Instruct yükleyici ve inference wrapper.
bitsandbytes NF4 4-bit quantization ile Colab T4'e sığar.
"""

import logging
from typing import Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name: str,
    load_in_4bit: bool = True,
    device_map: str = "auto",
    cache_dir: Optional[str] = None,
):
    """
    Modeli ve tokenizer'ı yükler.
    4-bit NF4 quantization varsayılan olarak aktif.
    """
    logger.info(f"Model yükleniyor: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        padding_side="left",    # batch generation için sol padding
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    model.eval()

    logger.info("Model yüklendi.")
    return model, tokenizer


class TranslationModel:
    """
    Qwen 2.5 Instruct için sohbet şablonu tabanlı inference wrapper.
    """

    def __init__(self, model, tokenizer, config: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config["model"]

    def _build_prompt(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        self,
        system: str,
        user: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Tek cümle için metin üretir."""
        prompt = self._build_prompt(system, user)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        max_tok = max_new_tokens or self.cfg["max_new_tokens_translation"]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tok,
                do_sample=self.cfg.get("do_sample", False),
                temperature=self.cfg.get("temperature", 1.0) if self.cfg.get("do_sample") else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Yalnızca üretilen kısmı decode et
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate_batch(
        self,
        prompts: list[tuple[str, str]],   # [(system, user), ...]
        max_new_tokens: Optional[int] = None,
        batch_size: int = 4,
    ) -> list[str]:
        """
        Birden fazla (system, user) çiftini batch halinde işler.
        Colab T4'te batch_size=2-4 önerilir.
        """
        max_tok = max_new_tokens or self.cfg["max_new_tokens_translation"]
        results = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i: i + batch_size]
            texts = [self._build_prompt(sys_, usr) for sys_, usr in batch]

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tok,
                    do_sample=self.cfg.get("do_sample", False),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            for out in output_ids:
                generated = out[input_len:]
                results.append(
                    self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                )

        return results
