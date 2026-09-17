"""Small, self-contained text corpora for deterministic synthetic rendering."""

from __future__ import annotations

from typing import Literal

import numpy as np


Split = Literal["train", "validation"]

TRAIN_WORDS = {
    "ru": (
        "авито", "доставка", "товар", "новый", "цена", "город", "улица",
        "магазин", "скидка", "заказ", "работа", "услуги", "ремонт", "дом",
        "квартира", "телефон", "мебель", "одежда", "книга", "подарок",
        "качество", "продажа", "покупка", "сегодня", "открыто", "вход",
    ),
    "en": (
        "delivery", "product", "new", "price", "city", "street", "market",
        "discount", "order", "service", "repair", "home", "phone", "office",
        "coffee", "fashion", "book", "gift", "quality", "sale", "open",
        "today", "online", "original", "premium", "center",
    ),
}

VALIDATION_WORDS = {
    "ru": (
        "объявление", "витрина", "распродажа", "контакты", "мастерская",
        "документы", "техника", "запчасти", "аренда", "склад", "выход",
        "внимание", "гарантия", "ежедневно", "подробности", "наличие",
    ),
    "en": (
        "classified", "showroom", "clearance", "contacts", "workshop",
        "documents", "equipment", "details", "rental", "warehouse", "exit",
        "attention", "warranty", "available", "information", "weekend",
    ),
}


def _choice(rng: np.random.Generator, values: tuple[str, ...]) -> str:
    return values[int(rng.integers(0, len(values)))]


def _words(split: Split, language: str) -> tuple[str, ...]:
    corpus = TRAIN_WORDS if split == "train" else VALIDATION_WORDS
    return corpus[language]


def generate_text(
    rng: np.random.Generator,
    split: Split,
    language: str,
    min_words: int,
    max_words: int,
    multiline_probability: float,
) -> str:
    """Generate a text string from split-specific words and templates."""
    word_counts = np.arange(min_words, max_words + 1)
    default_weights = np.array([0.30, 0.30, 0.20, 0.10, 0.07, 0.03])
    if min_words == 1 and max_words <= len(default_weights):
        weights = default_weights[:max_words]
    else:
        weights = np.exp(-0.55 * (word_counts - min_words))
    weights = weights / weights.sum()
    word_count = int(rng.choice(word_counts, p=weights))
    if language in {"ru", "en"}:
        tokens = [_choice(rng, _words(split, language)) for _ in range(word_count)]
    elif language == "digits":
        offset = 10_000 if split == "validation" else 0
        tokens = []
        for _ in range(word_count):
            kind = int(rng.integers(0, 4))
            value = int(rng.integers(offset, offset + 10_000))
            if kind == 0:
                tokens.append(f"{value / 100:.2f}")
            elif kind == 1:
                tokens.append(f"+7-{value:04d}-{int(rng.integers(100, 1000))}")
            elif kind == 2:
                tokens.append(f"{int(rng.integers(1, 29)):02d}.{int(rng.integers(1, 13)):02d}.2026")
            else:
                tokens.append(f"#{value:05d}")
    elif language == "mixed":
        tokens = []
        for _ in range(word_count):
            selected = "ru" if bool(rng.integers(0, 2)) else "en"
            tokens.append(_choice(rng, _words(split, selected)))
        insert_at = int(rng.integers(0, len(tokens) + 1))
        split_offset = 50_000 if split == "validation" else 20_000
        tokens.insert(insert_at, str(int(rng.integers(split_offset, split_offset + 10_000))))
    else:
        raise ValueError(f"Unsupported language: {language}")

    separator = _choice(rng, (" ", "  ", " · ", " | ", " — "))
    text = separator.join(tokens)
    effective_multiline_probability = (
        max(multiline_probability, 0.45) if len(tokens) >= 4 else multiline_probability
    )
    if len(tokens) > 1 and rng.random() < effective_multiline_probability:
        split_at = int(rng.integers(1, len(tokens)))
        text = separator.join(tokens[:split_at]) + "\n" + separator.join(tokens[split_at:])
    case = int(rng.integers(0, 4))
    if case == 1:
        text = text.upper()
    elif case == 2:
        text = text.title()
    return text
