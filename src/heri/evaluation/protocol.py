"""Versioned scientific reporting contract; independent of checkpoint architecture."""

EVALUATION_PROTOCOL = "heri_grounded"


def protocol_metadata(
    *,
    decision_threshold_source: str = "fixed_0.5",
    image_presence_source: str | None = "fixed_0.5",
) -> dict[str, str]:
    return {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "cf1_definition": "harmonic_mean_of_class_mean_precision_and_recall",
        "threshold_protocol": (
            "fixed_0.5_primary"
            if decision_threshold_source == "fixed_0.5"
            else (
                "validation_tuned_primary_and_fixed_0.5_secondary"
                if "validation_tuned" in str(decision_threshold_source)
                else f"{decision_threshold_source}_primary_and_fixed_0.5_secondary"
            )
        ),
        "text_f1_definition": "all_samples_native_subwords_visible_after_truncation",
        "text_word_f1_definition": "all_samples_original_words_missing_predictions_are_negative",
        "text_f1_positive_only_definition": "text_manipulated_samples_visible_words_max_over_subwords",
        "text_grounding_aggregation_definition": "eq21_probability_weighted_relative_mixture",
        "image_mean_iou_definition": "annotation_available_samples_original_coordinates_clipped_to_valid_image",
        "samm_released_all_sample_image_mean_iou_definition": "all_samples_released_empty_target_convention_epsilon_1e-4_with_HERA_preprocessing",
        "image_presence_gate_definition": (
            "eq22_any_individual_image_hypothesis_at_least_0.5_"
            + (image_presence_source or "disabled")
        ),
        "evidence_decision_consistency_definition": "eq32_mean_min_one_support_over_probability_plus_1e-8",
        "unsupported_decision_definition": "eq32_active_applicable_annotation_available_support_below_0.5",
        "verified_support_definition": "eq33_true_positive_hypothesis_top1_valid_position_support_at_least_0.5_and_inside_annotation",
        "evidence_decision_empty_domain_policy": "undefined_null_when_no_valid_active_sample_hypothesis_modality_units",
        "evidence_decision_applicability": "models_with_hypothesis_conditioned_per_modality_evidence_support_only",
        "samm_multilabel_protocol": "released_three_label_targets_fixed_0.5",
        "samm_reporting_protocol": "samm_released_v2_fixed05",
        "grounding_comparability": "native_tokenizer_and_preprocessing_not_identical_to_HAMMER",
    }
