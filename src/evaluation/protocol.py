"""Versioned scientific reporting contract; independent of checkpoint architecture."""

EVALUATION_PROTOCOL = "hera_scientific_v3"


def protocol_metadata() -> dict[str, str]:
    return {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "cf1_definition": "harmonic_mean_of_class_mean_precision_and_recall",
        "threshold_protocol": "validation_tuned_and_fixed_0.5_reported_separately",
        "text_f1_definition": "all_samples_native_subwords_visible_after_truncation",
        "text_word_f1_definition": "all_samples_original_words_missing_predictions_are_negative",
        "text_f1_positive_only_definition": "text_manipulated_samples_visible_words_max_over_subwords",
        "text_grounding_aggregation_definition": "target_free_configured_hypothesis_presence_aggregation",
        "image_mean_iou_definition": "all_samples_raw_boxes_empty_target_match_epsilon_1e-4",
        "grounding_comparability": "native_tokenizer_and_preprocessing_not_identical_to_HAMMER",
    }
