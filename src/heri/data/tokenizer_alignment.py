"""Word-level DGM4 grounding to subword-token alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .schema import DGM4SchemaError


class TokenizerAlignmentError(ValueError):
    """Raised when word-to-subword alignment cannot be produced."""


@dataclass(frozen=True)
class TokenAlignmentResult:
    """Subword grounding labels and mask produced from word-level labels."""

    labels: list[int]
    mask: list[int]
    word_ids: list[int | None]
    encoding: Any | None = None
    token_spans: list[tuple[int, int]] | None = None


def token_character_spans(words: Sequence[str], word_ids: Sequence[int | None],
                          offsets: Any | None = None) -> list[tuple[int, int]]:
    """Map pretokenized subword offsets into the shared ``' '.join(words)`` frame.

    Word-only test tokenizers have a whole-word fallback; real fast tokenizers
    provide their subword offsets. Special/padding tokens use (-1,-1).
    """
    starts: list[int] = []
    cursor = 0
    for word in words:
        starts.append(cursor)
        cursor += len(word) + 1
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    spans: list[tuple[int, int]] = []
    for index, word_id in enumerate(word_ids):
        if word_id is None or int(word_id) < 0:
            spans.append((-1, -1))
            continue
        wid = int(word_id)
        if wid >= len(words):
            raise TokenizerAlignmentError("Tokenizer word ID exceeds supplied words")
        start, end = (0, len(words[wid])) if offsets is None else offsets[index]
        spans.append((starts[wid] + int(start), starts[wid] + int(end)))
    return spans


def validate_word_positions(words: Sequence[str], fake_text_pos: Iterable[int]) -> set[int]:
    """Validate 0-based word positions and return them as a set."""

    positions = set(fake_text_pos)
    for position in positions:
        if not isinstance(position, int):
            raise DGM4SchemaError(f"fake_text_pos must contain ints, got {position!r}")
        if position < 0 or position >= len(words):
            raise DGM4SchemaError(
                f"fake_text_pos index {position} out of range for {len(words)} words"
            )
    return positions


def _word_ids_from_encoding(encoding: Any) -> list[int | None]:
    if hasattr(encoding, "word_ids"):
        try:
            return list(encoding.word_ids())
        except TypeError:
            return list(encoding.word_ids(batch_index=0))
    raise TokenizerAlignmentError(
        "Tokenizer encoding does not expose word_ids(); a fast tokenizer is required"
    )


def align_word_positions_to_subwords(
    *,
    words: Sequence[str],
    fake_text_pos: Sequence[int],
    tokenizer: Any | None = None,
    encoding: Any | None = None,
    max_length: int | None = None,
    ignore_index: int = -100,
) -> TokenAlignmentResult:
    """Map DGM4 word-level fake positions to tokenizer subword labels.

    Special tokens and padding positions receive ``ignore_index`` and mask 0.
    Real word subwords receive mask 1 and label 0/1.
    """

    positive_words = validate_word_positions(words, fake_text_pos)
    if encoding is None:
        if tokenizer is None:
            word_ids = [idx for idx, _ in enumerate(words)]
            labels = [1 if idx in positive_words else 0 for idx in word_ids]
            mask = [1 for _ in word_ids]
            return TokenAlignmentResult(labels=labels, mask=mask, word_ids=word_ids,
                                        token_spans=token_character_spans(words, word_ids))
        kwargs: dict[str, Any] = {
            "is_split_into_words": True,
            "return_attention_mask": True,
            "truncation": True,
            "return_offsets_mapping": True,
        }
        if max_length is not None:
            kwargs["max_length"] = max_length
            kwargs["padding"] = "max_length"
        encoding = tokenizer(list(words), **kwargs)

    word_ids = _word_ids_from_encoding(encoding)
    labels: list[int] = []
    mask: list[int] = []
    for word_id in word_ids:
        if word_id is None:
            labels.append(ignore_index)
            mask.append(0)
        else:
            labels.append(1 if word_id in positive_words else 0)
            mask.append(1)
    return TokenAlignmentResult(
        labels=labels,
        mask=mask,
        word_ids=word_ids,
        encoding=encoding,
        token_spans=token_character_spans(words, word_ids, encoding.get("offset_mapping")),
    )
