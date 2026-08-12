from src.utils.config import load_yaml_config, validate_baseline_config


def test_phase2_yaml_configs_validate() -> None:
    files = {
        "config/train_baseline_b0.yaml": "train",
        "config/train_baseline_b1.yaml": "train",
        "config/evaluate_baseline_b0.yaml": "evaluate",
        "config/evaluate_baseline_b1.yaml": "evaluate",
        "config/smoke_baseline_b0.yaml": "smoke",
        "config/smoke_baseline_b1.yaml": "smoke",
    }
    for path, mode in files.items():
        validate_baseline_config(load_yaml_config(path), mode=mode)
