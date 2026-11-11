"""Research comparisons describe actual operators and expose coupled changes."""

from __future__ import annotations

import copy
import json

import pytest

from heri.experiments.ablation import apply_hera_ablation_variant
from heri.experiments.cycle_study import apply_hera_cycle_study_variant
from heri.experiments.design import describe_model_design, design_delta
from heri.graph.reasoner import HypothesisEvidenceGraphReasoner
from heri.losses.heri_components import HERAMultitaskLoss
from heri.models.heri.decision import HERADecisionModule
from heri.routing import EvidenceReliabilityRouter
from heri.utils.config import load_yaml_config, validate_hera_config
from tests.support.paths import REPO_ROOT


def _canonical():
    return load_yaml_config(REPO_ROOT / "config/datasets/dgm4/heri/train.yaml")


def test_disabled_edge_attention_selects_standard_operator_and_reports_actual_design():
    config = apply_hera_ablation_variant(_canonical(), "no_router")
    config["graph"]["edge_attention"]["enabled"] = False
    reasoner, _, _, _ = _tiny_modules(config)
    assert reasoner.hgt.layers[0].variant == "standard"
    design = describe_model_design(config)
    assert design["graph"]["attention"] == "standard"
    assert design["graph"]["reliability_conditioned_attention"] is False


def _tiny_modules(config):
    config = copy.deepcopy(config)
    config["graph"].update(hidden_dim=4, layers=1, heads=2)
    config["model"]["hidden_dim"] = 4
    config["model"]["attribution"]["hidden_dim"] = 4
    config["routing"]["feature_dim"] = 4
    return (
        HypothesisEvidenceGraphReasoner.from_config(config),
        HERADecisionModule.from_config(config),
        EvidenceReliabilityRouter.from_config(config),
        HERAMultitaskLoss(config["loss"], config["routing"]),
    )


@pytest.mark.parametrize(
    ("variant", "readout", "graph_active", "router_active", "binary", "sufficiency_weight"),
    [
        ("evidence_decomposition", "family_mean_projection", False, False, "independent_head", 0.0),
        ("reliability_router", "family_mean_projection", False, True, "independent_head", 0.0),
        ("hypothesis_evidence_graph", "hypothesis_graph", True, True, "independent_head", 0.0),
        ("constrained_single_pass", "hypothesis_graph", True, True, "noisy_or", 0.1),
    ],
)
def test_progressive_description_matches_instantiated_operators_and_objectives(
    variant, readout, graph_active, router_active, binary, sufficiency_weight
):
    config = apply_hera_ablation_variant(_canonical(), variant)
    design = describe_model_design(config)
    reasoner, decision, router, criterion = _tiny_modules(config)
    assert design["model"]["readout"] == config["ablation"]["readout"] == readout
    assert config["ablation"]["experimental_design"] == design
    assert design["graph"]["message_passing"] == (not reasoner.pooled) == graph_active
    assert design["graph"]["hypothesis_nodes"] == reasoner.include_hypothesis_nodes == graph_active
    assert design["routing"]["enabled"] == router.enabled == router_active
    assert design["heads"]["binary_aggregation"] == decision.binary_aggregation == binary
    assert design["objectives"]["evidence_sufficiency"]["available"] is True
    assert (
        design["objectives"]["evidence_sufficiency"]["weight"]
        == criterion.lambda_ev
        == sufficiency_weight
    )
    assert (
        design["objectives"]["alignment"]["aggregation"]
        == config["loss"]["alignment"]["aggregation"]
    )
    assert design["reasoning"]["cycles"] == reasoner.cycles == 1
    assert not design["reasoning"]["dynamic_rebuild"]
    assert not design["graph"]["reliability_conditioned_attention"] or graph_active
    json.dumps(design, allow_nan=False)


def test_no_hypothesis_control_exposes_node_and_readout_intervention():
    canonical = _canonical()
    no_hypothesis = apply_hera_ablation_variant(canonical, "no_hypothesis_nodes")
    design = describe_model_design(no_hypothesis)
    reasoner, _, _, _ = _tiny_modules(no_hypothesis)
    assert design["model"]["readout"] == "family_mean_projection"
    assert design["graph"]["message_passing"] is True and not reasoner.pooled
    assert design["graph"]["hypothesis_nodes"] is False and not reasoner.include_hypothesis_nodes
    assert design["interpretation"]["coupled_interventions"] == [
        "hypothesis_graph_nodes",
        "prediction_readout",
    ]
    assert "does not isolate node count alone" in design["interpretation"]["comparison_limit"]
    assert describe_model_design(canonical)["model"]["readout"] == "hypothesis_graph"
    assert set(design_delta(canonical, no_hypothesis)) == {"graph.include_hypothesis_nodes"}


def test_static_cycle_control_preserves_readout_and_losses_while_freezing_refresh():
    canonical = _canonical()
    full = apply_hera_cycle_study_variant(canonical, "full_hera")
    static = apply_hera_cycle_study_variant(canonical, "two_cycle_static")
    full_design, static_design = map(describe_model_design, (full, static))
    assert full_design["model"] == static_design["model"]
    assert full_design["heads"] == static_design["heads"]
    assert full_design["objectives"] == static_design["objectives"]
    assert full_design["reasoning"]["cycles"] == static_design["reasoning"]["cycles"] == 2
    assert full_design["reasoning"]["refresh_semantic"] is True
    assert static_design["reasoning"]["refresh_semantic"] is False
    assert set(design_delta(full, static)) == {
        "graph.reasoning.dynamic_rebuild",
        "graph.reasoning.refresh_hypothesis",
        "graph.reasoning.refresh_semantic",
    }


@pytest.mark.parametrize(
    ("baseline", "readout"), [("b0", "global_fusion"), ("b1", "cross_attention")]
)
def test_baselines_do_not_claim_graph_evidence_or_sufficiency(baseline, readout):
    config = load_yaml_config(REPO_ROOT / f"config/datasets/dgm4/baselines/{baseline}/train.yaml")
    design = describe_model_design(config)
    assert design["model"]["readout"] == readout
    assert design["evidence"]["families"] == []
    assert design["graph"]["message_passing"] is False
    assert design["routing"]["enabled"] is False
    assert design["objectives"]["evidence_sufficiency"]["available"] is False
    assert design["objectives"]["alignment"]["available"] is False


def test_learned_queries_are_not_reported_as_hypothesis_graph_nodes():
    config = _canonical()
    config["reasoning_control"] = {"enabled": True, "variant": "learned_queries"}
    design = describe_model_design(config)
    assert design["model"]["readout"] == "learned_class_queries"
    assert design["graph"]["message_passing"] is False
    assert design["graph"]["hypothesis_nodes"] is False
    assert not design["reasoning"]["refresh_hypothesis"]


def test_design_delta_ignores_seed_location_resume_and_runtime_bookkeeping():
    before = _canonical()
    after = copy.deepcopy(before)
    after["project"].update(seed=21, name="another-name", config_name="another-config")
    after["dataloader"].update(seed=21, num_workers=0)
    after["dataset"].update(root="elsewhere", train_annotations="elsewhere/train.json")
    after["output"]["root"] = "another-output"
    after["training"]["resume_from"] = "another-checkpoint.pt"
    after["training"]["auto_resume"]["enabled"] = False
    after["runtime"].update(device="cpu", cuda_memory_logging={"enabled": False})
    assert design_delta(before, after) == {}


def test_design_delta_records_stable_scientific_changes_and_detaches_values():
    before = _canonical()
    after = copy.deepcopy(before)
    after["loss"]["lambda_ev"] = 0.0
    after["graph"]["heads"] = 4
    after["runtime"]["precision"] = "fp32"
    after["model"]["hypotheses"]["order"] = ["TS", "TA", "FS", "FA"]
    after["loss"]["custom_setting"] = None
    del after["scheduler"]["resume_horizon_policy"]
    delta = design_delta(before, after)
    assert list(delta) == sorted(delta)
    assert delta["loss.lambda_ev"] == {"before": 0.1, "after": 0.0, "change": "changed"}
    assert delta["loss.custom_setting"] == {"before": None, "after": None, "change": "added"}
    assert delta["scheduler.resume_horizon_policy"]["change"] == "removed"
    assert delta["runtime.precision"]["after"] == "fp32"
    delta["model.hypotheses.order"]["after"].append("OTHER")
    assert after["model"]["hypotheses"]["order"] == ["TS", "TA", "FS", "FA"]


def test_router_removal_preserves_unrelated_evidence_and_alignment_weights():
    before = _canonical()
    before["loss"].update(lambda_ev=0.3, lambda_align=0.2)
    before["routing"]["regularization"]["weight"] = 0.075
    after = apply_hera_ablation_variant(before, "no_router")
    assert after["loss"]["lambda_ev"] == 0.3
    assert after["loss"]["lambda_align"] == 0.2
    assert after["routing"]["regularization"]["weight"] == 0.0
    assert set(design_delta(before, after)) == {
        "routing.enabled",
        "routing.regularization.enabled",
        "routing.regularization.weight",
    }


def test_forensic_removal_preserves_configured_regularizer_and_objective_weights():
    before = _canonical()
    before["loss"].update(lambda_ev=0.3, lambda_align=0.2)
    before["routing"]["regularization"]["weight"] = 0.075
    after = apply_hera_ablation_variant(before, "no_forensic_evidence")
    assert after["routing"]["regularization"]["weight"] == 0.075
    assert after["loss"]["lambda_ev"] == 0.3
    assert after["loss"]["lambda_align"] == 0.2
    assert set(design_delta(before, after)) == {"evidence.visual_forensic.enabled"}


@pytest.mark.parametrize(
    "variant", ["evidence_decomposition", "reliability_router", "hypothesis_evidence_graph"]
)
def test_unconstrained_progressive_stages_deliberately_disable_sufficiency(variant):
    before = _canonical()
    before["loss"].update(lambda_ev=0.3, lambda_align=0.2)
    after = apply_hera_ablation_variant(before, variant)
    assert after["loss"]["lambda_ev"] == 0.0
    assert after["loss"]["lambda_align"] == 0.2
    assert (
        after["ablation"]["experimental_design"]["objectives"]["evidence_sufficiency"]["enabled"]
        is False
    )


def test_constrained_progressive_stage_retains_positive_base_sufficiency_weight():
    before = _canonical()
    before["loss"].update(lambda_ev=0.3, lambda_align=0.2)
    after = apply_hera_ablation_variant(before, "constrained_single_pass")
    assert after["loss"]["lambda_ev"] == 0.3
    assert after["loss"]["lambda_align"] == 0.2
    assert (
        after["ablation"]["experimental_design"]["objectives"]["evidence_sufficiency"]["weight"]
        == 0.3
    )


@pytest.mark.parametrize("weight", [0.0, -0.1, float("nan")])
def test_constrained_progressive_stage_rejects_nonpositive_or_nonfinite_sufficiency(weight):
    before = _canonical()
    before["loss"]["lambda_ev"] = weight
    with pytest.raises(ValueError, match="positive loss.lambda_ev"):
        apply_hera_ablation_variant(before, "constrained_single_pass")


def test_router_removal_preserves_three_cycle_static_base_and_other_graph_choices():
    before = _canonical()
    before["graph"]["reasoning"].update(
        cycles=3,
        dynamic_rebuild=False,
        refresh_semantic=False,
        refresh_hypothesis=False,
        hypothesis_state="initial",
        hypothesis_feedback=False,
    )
    before["graph"]["edge_attention"]["variant"] = "standard"
    before["graph"]["edges"]["feedback"]["enabled"] = False
    before["model"]["grounding"]["mode"] = "generic"
    after = apply_hera_ablation_variant(before, "no_router")
    assert after["graph"] == before["graph"]
    assert after["model"] == before["model"]
    assert after["loss"] == before["loss"]
    validate_hera_config(after, mode="train")
    assert set(design_delta(before, after)) == {
        "routing.enabled",
        "routing.regularization.enabled",
        "routing.regularization.weight",
    }


def test_forensic_control_rejects_unsupported_independent_base_without_rewriting_it():
    before = _canonical()
    before["model"]["binary"]["aggregation"] = "independent_head"
    untouched = copy.deepcopy(before)
    with pytest.raises(ValueError, match="independent-head base is not supported"):
        apply_hera_ablation_variant(before, "no_forensic_evidence")
    assert before == untouched


def test_different_component_controls_cannot_be_silently_layered():
    no_router = apply_hera_ablation_variant(_canonical(), "no_router")
    with pytest.raises(ValueError, match="cannot be layered"):
        apply_hera_ablation_variant(no_router, "no_forensic_evidence")
    assert apply_hera_ablation_variant(no_router, "no_router") == no_router


@pytest.mark.parametrize("variant", ["hypothesis_evidence_graph", "constrained_single_pass"])
def test_graph_progressive_stages_explicitly_activate_hypothesis_graph_readout(variant):
    before = _canonical()
    before["graph"]["reasoning"]["operator"] = "pooled"
    before["graph"]["include_hypothesis_nodes"] = False
    after = apply_hera_ablation_variant(before, variant)
    assert after["graph"]["reasoning"]["operator"] == "graph"
    assert after["graph"]["include_hypothesis_nodes"] is True
    assert after["ablation"]["readout"] == "hypothesis_graph"
