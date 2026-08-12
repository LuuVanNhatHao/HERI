"""Run the cumulative HERA-DGM deferred server validation gate."""

from __future__ import annotations

import json
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
    iterator = tqdm(phases, desc="Server validation phases", disable=not bool(config["progress"]["enabled"]))
    for phase in iterator:
        phase_result = _run_phase(phase, output_root=output_root, progress=bool(config["progress"]["enabled"]))
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
        "canonical_training_allowed": final_status == "PASS",
    }
    (output_root / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({"status": final_status, "canonical_training_allowed": final_status == "PASS"}, indent=2))
    return 0 if final_status == "PASS" else 1


def _run_phase(phase: Mapping[str, Any], *, output_root: Path, progress: bool) -> dict[str, Any]:
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
        results.append(_run_check(check, output_root=output_root))
    required_failed = any(item["required_for_training"] and item["status"] != "PASS" for item in results)
    return {
        "phase": phase["name"],
        "status": "FAIL" if required_failed else "PASS",
        "checks": results,
    }


def _run_check(check: Mapping[str, Any], *, output_root: Path) -> dict[str, Any]:
    command = [str(part) for part in check["command"]]
    if command[0] == "python":
        command[0] = sys.executable
    started = time.perf_counter()
    timeout = float(check.get("timeout_seconds", 0)) or None
    log_name = str(check.get("log_file", f"{check['name']}.json")).replace("/", "_").replace("\\", "_")
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
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
        "runtime_seconds": time.perf_counter() - started,
        "stdout_tail": "".join(stdout_tail),
        "stderr_tail": "".join(stderr_tail),
        "stream_log": str(stream_log_path),
        "expected_pass_criteria": check.get("expected_pass_criteria", []),
    }
    (logs_dir / log_name).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


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
