"""Materialize first; execute only explicitly selected GPU jobs with --run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.scientific_study import aggregate_study, digest, materialize_study, source_digest
from src.utils.config import dump_yaml, load_yaml_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", default="config/experiments/dgm4/scientific_v3.yaml"
    )
    parser.add_argument("--manifest", help="Reuse an existing manifest to aggregate or resume identical code")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted job from its own last.pt")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--phase", choices=["all", "train", "evaluate"], default="all")
    args = parser.parse_args()
    manifest = (json.loads(Path(args.manifest).read_text(encoding="utf-8")) if args.manifest
                else materialize_study(load_yaml_config(ROOT / args.suite)["suite"], ROOT))
    manifest_path = Path(args.manifest or manifest["manifest_path"])
    if args.run:
        if source_digest(ROOT) != manifest["source_digest"]:
            raise ValueError("Source changed; materialize a new study instead of mixing runs")
        known_variants = {job["variant"] for job in manifest["jobs"]}
        if set(args.variants or []) - known_variants or set(args.seeds or []) - set(manifest["seeds"]):
            raise ValueError("Requested variant/seed is absent from manifest")
        for job in manifest["jobs"]:
            if args.variants and job["variant"] not in args.variants or args.seeds and job["seed"] not in args.seeds:
                continue
            for phase in (["train", "evaluate"] if args.phase == "all" else [args.phase]):
                config_path = Path(job[f"{phase}_config"])
                if digest(load_yaml_config(config_path)) != job[f"{phase}_config_hash"]:
                    raise ValueError(f"Materialized config changed: {config_path}")
                marker = config_path.parent / f"{phase}_completed.json"
                if marker.is_file():
                    record = json.loads(marker.read_text(encoding="utf-8"))
                    if record != {"protocol_id": manifest["protocol_id"], "config_hash": job[f"{phase}_config_hash"]}:
                        raise ValueError(f"Invalid completion marker: {marker}")
                    artifact = Path(job["summary"]) if phase == "evaluate" else config_path.parent.parent / "train/checkpoints/best.pt"
                    if not artifact.is_file():
                        raise FileNotFoundError(f"Completed job lost artifact: {artifact}")
                    continue
                if phase == "train" and (config_path.parent.parent / "train").exists():
                    last = config_path.parent.parent / "train/checkpoints/last.pt"
                    if not args.resume or not last.is_file():
                        raise ValueError(f"Existing incomplete run: {last.parent.parent}. Use --resume with its last.pt or a new study output_root")
                    resumed = load_yaml_config(config_path)
                    resumed["training"]["resume_from"] = str(last)
                    config_path = config_path.parent / "resume.yaml"
                    dump_yaml(resumed, config_path)
                command = [sys.executable, str(ROOT / f"scripts/{phase}.py"), str(config_path)]
                print(f"{job['variant']} seed={job['seed']} {phase}", flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
                artifact = Path(job["summary"]) if phase == "evaluate" else config_path.parent.parent / "train/checkpoints/best.pt"
                if not artifact.is_file():
                    raise FileNotFoundError(f"Command exited without expected artifact: {artifact}")
                marker.write_text(json.dumps({"protocol_id": manifest["protocol_id"], "config_hash": job[f"{phase}_config_hash"]}), encoding="utf-8")
    report = aggregate_study(manifest)
    report_path = manifest_path.parent / "paper_results.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "paper_results": str(report_path), "status": report["status"], "pending_jobs": len(report["pending_jobs"])}, indent=2))


if __name__ == "__main__":
    main()
