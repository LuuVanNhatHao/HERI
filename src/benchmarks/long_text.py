"""Deterministic token-budget policies for long benchmark documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class LongTextPolicy:
    """Headline-preserving lead/tail selection within a model token budget."""

    max_length: int
    headline_max_tokens: int
    article_lead_fraction: float
    strategy: str = "headline_lead_tail_token_budget_v1"

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        max_length: int,
    ) -> "LongTextPolicy":
        return cls(
            max_length=int(max_length),
            headline_max_tokens=int(config["headline_max_tokens"]),
            article_lead_fraction=float(config["article_lead_fraction"]),
            strategy=str(config["strategy"]),
        )

    def validate(self) -> None:
        if self.strategy != "headline_lead_tail_token_budget_v1":
            raise ValueError(f"Unsupported long-text strategy: {self.strategy!r}")
        if self.max_length <= 4:
            raise ValueError("Long-text max_length must leave room for content")
        if not 0 < self.headline_max_tokens < self.max_length:
            raise ValueError("headline_max_tokens must be in (0,max_length)")
        if not 0 < self.article_lead_fraction < 1:
            raise ValueError("article_lead_fraction must be in (0,1)")


def encode_headline_article(
    tokenizer: Any,
    *,
    headline: str,
    article: str,
    policy: LongTextPolicy,
) -> dict[str, Any]:
    """Encode a pair while retaining the headline and both article boundaries.

    The function operates on tokenizer IDs, not characters or whitespace words,
    so the configured budget is exact for the selected tokenizer. Added special
    tokens receive word ID ``-1``; selected article word IDs retain their original
    positions, including the tail after middle truncation.
    """

    policy.validate()
    headline = str(headline).strip()
    article = str(article).strip()
    if not headline or not article:
        raise ValueError("Long-text encoding requires non-empty headline and article")

    headline_encoding = _tokenize_without_special_tokens(tokenizer, headline)
    article_encoding = _tokenize_without_special_tokens(tokenizer, article)
    headline_ids = headline_encoding["input_ids"]
    headline_word_ids = headline_encoding["word_ids"]
    article_ids = article_encoding["input_ids"]
    article_word_ids = article_encoding["word_ids"]

    special_tokens = _pair_special_token_count(tokenizer)
    content_budget = policy.max_length - special_tokens
    if content_budget <= 1:
        raise ValueError(
            "Tokenizer pair special tokens consume the configured long-text budget"
        )

    kept_headline_count = min(
        len(headline_ids), policy.headline_max_tokens, content_budget - 1
    )
    kept_headline_ids = headline_ids[:kept_headline_count]
    kept_headline_word_ids = headline_word_ids[:kept_headline_count]
    article_budget = content_budget - kept_headline_count
    if article_budget <= 0:
        raise ValueError("Long-text policy left no token budget for the article")

    if len(article_ids) <= article_budget:
        lead_count = len(article_ids)
        tail_count = 0
        kept_article_ids = article_ids
        kept_article_word_ids = article_word_ids
    else:
        lead_count = max(1, int(article_budget * policy.article_lead_fraction))
        lead_count = min(lead_count, article_budget - 1)
        tail_count = article_budget - lead_count
        kept_article_ids = article_ids[:lead_count] + article_ids[-tail_count:]
        kept_article_word_ids = (
            article_word_ids[:lead_count] + article_word_ids[-tail_count:]
        )

    prepared = tokenizer.prepare_for_model(
        kept_headline_ids,
        pair_ids=kept_article_ids,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=True,
        return_special_tokens_mask=False,
    )
    input_ids = _to_int_list(prepared["input_ids"])
    attention_mask = _to_int_list(
        prepared.get("attention_mask", [1] * len(input_ids))
    )
    special_mask_method = getattr(tokenizer, "get_special_tokens_mask", None)
    if not callable(special_mask_method):
        raise TypeError("Long-text tokenizer must expose get_special_tokens_mask")
    special_mask = _to_int_list(
        special_mask_method(input_ids, already_has_special_tokens=True)
    )
    if not (len(input_ids) == len(attention_mask) == len(special_mask)):
        raise ValueError("Prepared long-text tensors have inconsistent lengths")
    if len(input_ids) > policy.max_length:
        raise ValueError(
            f"Prepared long text exceeds max_length: {len(input_ids)} > {policy.max_length}"
        )

    headline_word_offset = _word_count(kept_headline_word_ids)
    shifted_article_word_ids = [
        -1 if value < 0 else value + headline_word_offset
        for value in kept_article_word_ids
    ]
    content_word_ids = iter(kept_headline_word_ids + shifted_article_word_ids)
    word_ids: list[int] = []
    for is_special in special_mask:
        word_ids.append(-1 if is_special else next(content_word_ids))
    try:
        next(content_word_ids)
    except StopIteration:
        pass
    else:
        raise ValueError("Tokenizer special-token mask did not consume all content IDs")

    original_token_count = len(headline_ids) + len(article_ids) + special_tokens
    retained_token_count = len(input_ids)
    truncated = (
        kept_headline_count < len(headline_ids)
        or len(kept_article_ids) < len(article_ids)
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "word_ids": word_ids,
        "grounding_mask": [
            bool(mask) and word_id >= 0
            for mask, word_id in zip(attention_mask, word_ids)
        ],
        "audit": {
            "strategy": policy.strategy,
            "max_length": policy.max_length,
            "original_token_count": original_token_count,
            "retained_token_count": retained_token_count,
            "retention_ratio": retained_token_count / max(1, original_token_count),
            "truncated": truncated,
            "headline_original_tokens": len(headline_ids),
            "headline_retained_tokens": kept_headline_count,
            "article_original_tokens": len(article_ids),
            "article_retained_tokens": len(kept_article_ids),
            "article_lead_retained_tokens": lead_count,
            "article_tail_retained_tokens": tail_count,
        },
    }


def summarize_long_text_lengths(
    tokenizer: Any,
    *,
    headlines: Sequence[str],
    articles: Sequence[str],
    policy: LongTextPolicy,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Return dataset-level token retention diagnostics without model execution."""

    policy.validate()
    if len(headlines) != len(articles):
        raise ValueError("Headline/article audit inputs must have equal length")
    if batch_size <= 0:
        raise ValueError("Long-text audit batch_size must be positive")
    original_lengths: list[int] = []
    retained_lengths: list[int] = []
    special_tokens = _pair_special_token_count(tokenizer)
    for start in range(0, len(headlines), batch_size):
        headline_batch = [str(value).strip() for value in headlines[start : start + batch_size]]
        article_batch = [str(value).strip() for value in articles[start : start + batch_size]]
        if any(not value for value in headline_batch + article_batch):
            raise ValueError("Long-text audit encountered empty headline/article")
        headline_ids = tokenizer(
            headline_batch,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        article_ids = tokenizer(
            article_batch,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        for headline_row, article_row in zip(headline_ids, article_ids):
            original = len(headline_row) + len(article_row) + special_tokens
            headline_retained = min(
                len(headline_row),
                policy.headline_max_tokens,
                policy.max_length - special_tokens - 1,
            )
            article_budget = policy.max_length - special_tokens - headline_retained
            retained = special_tokens + headline_retained + min(
                len(article_row), article_budget
            )
            original_lengths.append(original)
            retained_lengths.append(retained)

    truncated_count = sum(
        retained < original
        for original, retained in zip(original_lengths, retained_lengths)
    )
    ratios = [
        retained / max(1, original)
        for original, retained in zip(original_lengths, retained_lengths)
    ]
    return {
        "strategy": policy.strategy,
        "max_length": policy.max_length,
        "samples": len(original_lengths),
        "truncated_samples": truncated_count,
        "truncated_fraction": truncated_count / max(1, len(original_lengths)),
        "mean_retention_ratio": sum(ratios) / max(1, len(ratios)),
        "original_tokens": _length_statistics(original_lengths),
        "retained_tokens": _length_statistics(retained_lengths),
    }


def _tokenize_without_special_tokens(tokenizer: Any, text: str) -> dict[str, list[int]]:
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )
    input_ids = _to_int_list(encoding["input_ids"])
    word_ids_method = getattr(encoding, "word_ids", None)
    if not callable(word_ids_method):
        raise TypeError("Long-text policy requires a fast tokenizer with word_ids()")
    raw_word_ids = word_ids_method()
    if len(raw_word_ids) != len(input_ids):
        raise ValueError("Long-text tokenizer word IDs do not align with input IDs")
    return {
        "input_ids": input_ids,
        "word_ids": [-1 if value is None else int(value) for value in raw_word_ids],
    }


def _pair_special_token_count(tokenizer: Any) -> int:
    method = getattr(tokenizer, "num_special_tokens_to_add", None)
    if not callable(method):
        raise TypeError("Long-text tokenizer must expose num_special_tokens_to_add")
    count = int(method(pair=True))
    if count <= 0:
        raise ValueError("Long-text tokenizer reported no pair special tokens")
    return count


def _word_count(word_ids: Sequence[int]) -> int:
    valid = [int(value) for value in word_ids if int(value) >= 0]
    return max(valid, default=-1) + 1


def _to_int_list(values: Any) -> list[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(value) for value in values]


def _length_statistics(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "mean": 0.0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    ordered = sorted(int(value) for value in values)

    def percentile(fraction: float) -> int:
        index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
        return ordered[index]

    return {
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }
