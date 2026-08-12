"""Run the cumulative HERA-DGM deferred server validation gate."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import dump_yaml, load_yaml_config, validate_server_validation_config
from src.utils.environment import collect_environment_metadata
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/server_validate.py <config.yaml>", file=sys.stderr)
        return 2
    config = load_yaml_config(sys.argv[1])
    validate_server_validation_config(config)
    setup_logging(str(config.get("logging", {}).get("level", "INFO")))
    set_seed(int(config["project"]["seed"]))
    output_root = Path(str(config["output"]["root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    dump_yaml(config, output_root / "effective_config.yaml")
    (output_root / "environment.json").write_text(
        json.dumps(collect_environment_metadata(ROOT), indent=2, default=str),
        encoding="utf-8",
    )

    started = time.perf_counter()
    phase_results: list[dict[str, Any]] = []
    phases = config["validation"]["phases"]
    validation_settings = config["validation"]
    git_commit = _current_git_commit()
    iterator = tqdm(phases, desc="Server validation phases", disable=not bool(config["progress"]["enabled"]))
    for phase in iterator:
        phase_result = _run_phase(
            phase,
            output_root=output_root,
            progress=bool(config["progress"]["enabled"]),
            validation_settings=validation_settings,
            git_commit=git_commit,
        )
        phase_results.append(phase_result)
        (output_root / str(phase["output_file"])).write_text(
            json.dumps(phase_result, indent=2, default=str),
            encoding="utf-8",
        )
        if phase_result["status"] == "FAIL" and bool(config["validation"].get("stop_on_failure", True)):
            break
    required_failures = [
        check
        for phase in phase_results
        for check in phase["checks"]
        if check["required_for_training"] and check["status"] != "PASS"
    ]
    final_status = "PASS" if not required_failures else "FAIL"
    summary = {
        "project": config["project"]["name"],
        "record_type": "SERVER_VALIDATION_GATE",
        "status": final_status,
        "runtime_seconds": time.perf_counter() - started,
        "required_failures": required_failures,
        "phases": phase_results,
        "resume": {
            "resume_completed_checks": bool(validation_settings["resume_completed_checks"]),
            "force_rerun_all": bool(validation_settings["force_rerun_all"]),
            "git_commit": git_commit,
        },
        "canonical_training_allowed": final_status == "PASS",
    }
    (output_root / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({"status": final_status, "canonical_training_allowed": final_status == "PASS"}, indent=2))
    return 0 if final_status == "PASS" else 1


def _run_phase(
    phase: Mapping[str, Any],
    *,
    output_root: Path,
    progress: bool,
    validation_settings: Mapping[str, Any],
    git_commit: str | None,
) -> dict[str, Any]:
    checks = phase["checks"]
    results = []
    iterator = tqdm(
        checks,
        desc=str(phase["name"]),
        disable=not progress,
        leave=False,
    )
    for check in iterator:
        iterator.set_postfix(check=str(check["name"]))
        results.append(
            _run_check(
                check,
                output_root=output_root,
                validation_settings=validation_settings,
                git_commit=git_commit,
            )
        )
    required_failed = any(item["required_for_training"] and item["status"] != "PASS" for item in results)
    return {
        "phase": phase["name"],
        "status": "FAIL" if required_failed else "PASS",
        "checks": results,
    }


def _run_check(
    check: Mapping[str, Any],
    *,
    output_root: Path,
    validation_settings: Mapping[str, Any],
    git_commit: str | None,
) -> dict[str, Any]:
    command = [str(part) for part in check["command"]]
    started = time.perf_counter()
    timeout = float(check.get("timeout_seconds", 0)) or None
    log_name = _check_log_name(check)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    record_path = logs_dir / log_name
    metadata = {
        "git_commit": git_commit,
        "command_config_sha256": _command_config_sha256(command),
    }
    reusable = _load_reusable_check_record(
        check,
        record_path=record_path,
        validation_settings=validation_settings,
        metadata=metadata,
    )
    if reusable is not None:
        return reusable
    if command[0] == "python":
        command[0] = sys.executable
    stream_log_path = logs_dir / f"{Path(log_name).stem}.stream.log"
    tqdm.write(f"Running server validation check: {check['name']}")
    tqdm.write(f"Command: {' '.join(command)}")

    env = os.environ.copy()
    if command[0] == sys.executable:
        env["PYTHONUNBUFFERED"] = "1"
    stdout_tail: deque[str] = deque(maxlen=4000)
    stderr_tail: deque[str] = deque(maxlen=4000)
    lock = threading.Lock()
    timed_out = False
    with stream_log_path.open("w", encoding="utf-8", errors="replace") as stream_log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            env=env,
        )
        threads = [
            threading.Thread(
                target=_stream_pipe,
                args=(process.stdout, sys.stdout, stream_log, stdout_tail, lock),
                daemon=True,
            ),
            threading.Thread(
                target=_stream_pipe,
                args=(process.stderr, sys.stderr, stream_log, stderr_tail, lock),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            returncode = process.wait()
        for thread in threads:
            thread.join(timeout=2)
    record = {
        "name": check["name"],
        "command": check["command"],
        "required_for_training": bool(check["required_for_training"]),
        "status": "PASS" if returncode == 0 and not timed_out else "FAIL",
        "returncode": returncode,
        "timed_out": timed_out,
        "skipped": False,
        "runtime_seconds": time.perf_counter() - started,
        "stdout_tail": "".join(stdout_tail),
        "stderr_tail": "".join(stderr_tail),
        "stream_log": str(stream_log_path),
        **metadata,
        "expected_pass_criteria": check.get("expected_pass_criteria", []),
    }
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _check_log_name(check: Mapping[str, Any]) -> str:
    return str(check.get("log_file", f"{check['name']}.json")).replace("/", "_").replace("\\", "_")


def _load_reusable_check_record(
    check: Mapping[str, Any],
    *,
    record_path: Path,
    validation_settings: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    if bool(validation_settings["force_rerun_all"]) or not bool(validation_settings["resume_completed_checks"]):
        return None
    if not record_path.exists():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if record.get("status") != "PASS":
        return None
    if bool(validation_settings["resume_require_command_match"]) and record.get("command") != check["command"]:
        return None
    if bool(validation_settings["resume_require_config_hash_match"]):
        if record.get("command_config_sha256") != metadata["command_config_sha256"]:
            return None
    if bool(validation_settings["resume_require_git_commit_match"]):
        if record.get("git_commit") != metadata["git_commit"]:
            return None
    reusable = dict(record)
    reusable["skipped"] = True
    reusable["resume_source"] = str(record_path)
    tqdm.write(f"Skipping already passed server validation check: {check['name']}")
    return reusable


def _command_config_sha256(command: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for part in command:
        if not part.endswith((".yaml", ".yml")):
            continue
        path = ROOT / part
        if path.exists():
            hashes[part] = _sha256_file(path)
    return hashes


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _stream_pipe(
    pipe: Any,
    terminal: Any,
    stream_log: Any,
    tail: deque[str],
    lock: threading.Lock,
) -> None:
    if pipe is None:
        return
    try:
        while True:
            chunk = pipe.read(1)
            if not chunk:
                break
            with lock:
                tail.append(chunk)
                stream_log.write(chunk)
                terminal.write(chunk)
                stream_log.flush()
                terminal.flush()
    finally:
        pipe.close()


if __name__ == "__main__":
    raise SystemExit(main())
