from __future__ import annotations

import pytest

from src.benchmarks.long_text import (
    LongTextPolicy,
    encode_headline_article,
    summarize_long_text_lengths,
)


class _Encoding(dict):
    def __init__(self, input_ids: list[int], word_ids: list[int]) -> None:
        super().__init__(input_ids=input_ids)
        self._word_ids = word_ids

    def word_ids(self) -> list[int]:
        return list(self._word_ids)


class _WhitespacePairTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}

    def __call__(self, text, **_kwargs):
        if isinstance(text, list):
            return {"input_ids": [self._ids(value) for value in text]}
        ids = self._ids(text)
        return _Encoding(ids, list(range(len(ids))))

    def _ids(self, text: str) -> list[int]:
        ids: list[int] = []
        for token in str(text).split():
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary) + 1000
            ids.append(self.vocabulary[token])
        return ids

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        return 3 if pair else 2

    def prepare_for_model(self, ids, *, pair_ids, **_kwargs):
        input_ids = [101, *ids, 102, *pair_ids, 102]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "special_tokens_mask": [1, *([0] * len(ids)), 1, *([0] * len(pair_ids)), 1],
        }


def _policy(max_length: int = 20) -> LongTextPolicy:
    return LongTextPolicy(
        max_length=max_length,
        headline_max_tokens=4,
        article_lead_fraction=0.67,
    )


def test_long_text_policy_preserves_headline_and_article_lead_tail() -> None:
    tokenizer = _WhitespacePairTokenizer()
    headline = " ".join(f"h{index}" for index in range(6))
    article_tokens = [f"a{index}" for index in range(30)]
    encoded = encode_headline_article(
        tokenizer,
        headline=headline,
        article=" ".join(article_tokens),
        policy=_policy(),
    )

    assert len(encoded["input_ids"]) == 20
    audit = encoded["audit"]
    assert audit["truncated"] is True
    assert audit["headline_original_tokens"] == 6
    assert audit["headline_retained_tokens"] == 4
    assert audit["article_retained_tokens"] == 13
    assert audit["article_lead_retained_tokens"] == 8
    assert audit["article_tail_retained_tokens"] == 5

    expected_article = tokenizer._ids(" ".join(article_tokens[:8] + article_tokens[-5:]))
    assert encoded["input_ids"] == [
        101,
        *tokenizer._ids("h0 h1 h2 h3"),
        102,
        *expected_article,
        102,
    ]
    assert encoded["word_ids"][:6] == [-1, 0, 1, 2, 3, -1]
    assert encoded["word_ids"][-6:] == [29, 30, 31, 32, 33, -1]
    assert encoded["grounding_mask"] == [
        word_id >= 0 for word_id in encoded["word_ids"]
    ]


def test_short_long_context_record_is_not_padded_or_truncated() -> None:
    encoded = encode_headline_article(
        _WhitespacePairTokenizer(),
        headline="short headline",
        article="complete short article",
        policy=_policy(max_length=32),
    )

    assert len(encoded["input_ids"]) == 8
    assert encoded["audit"]["truncated"] is False
    assert encoded["audit"]["retention_ratio"] == pytest.approx(1.0)


def test_long_text_length_audit_reports_real_retention() -> None:
    report = summarize_long_text_lengths(
        _WhitespacePairTokenizer(),
        headlines=["h0 h1", "h0 h1 h2 h3 h4 h5"],
        articles=["a0 a1", " ".join(f"a{index}" for index in range(30))],
        policy=_policy(),
        batch_size=2,
    )

    assert report["samples"] == 2
    assert report["truncated_samples"] == 1
    assert report["truncated_fraction"] == pytest.approx(0.5)
    assert report["retained_tokens"]["max"] == 20

