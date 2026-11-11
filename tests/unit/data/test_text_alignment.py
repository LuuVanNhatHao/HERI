import pytest

from heri.data.tokenizer_alignment import align_word_positions_to_subwords


class FakeEncoding(dict):
    def __init__(self) -> None:
        super().__init__(
            {
                "input_ids": [101, 10, 11, 20, 30, 31, 102, 0],
                "attention_mask": [1, 1, 1, 1, 1, 1, 1, 0],
            }
        )

    def word_ids(self):
        return [None, 0, 0, 1, 2, 2, None, None]


class FakeTokenizer:
    is_fast = True

    def __call__(self, words, **kwargs):
        assert kwargs["is_split_into_words"] is True
        return FakeEncoding()


def test_0_based_word_positions_to_subwords() -> None:
    result = align_word_positions_to_subwords(
        words=["unbelievable", "middle", "token"],
        fake_text_pos=[0, 2],
        tokenizer=FakeTokenizer(),
        max_length=8,
    )
    assert result.labels == [-100, 1, 1, 0, 1, 1, -100, -100]
    assert result.mask == [0, 1, 1, 1, 1, 1, 0, 0]


def test_no_manipulation_labels_real_words_negative() -> None:
    result = align_word_positions_to_subwords(
        words=["a", "b", "c"],
        fake_text_pos=[],
        encoding=FakeEncoding(),
    )
    assert result.labels[1:6] == [0, 0, 0, 0, 0]


def test_invalid_word_position_detected() -> None:
    with pytest.raises(Exception):
        align_word_positions_to_subwords(words=["only"], fake_text_pos=[1])
