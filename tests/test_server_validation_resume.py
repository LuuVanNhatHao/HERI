from __future__ import annotations

import json

from scripts.server_validate import _run_check


def _resume_settings() -> dict:
    return {
        "resume_completed_checks": True,
        "resume_require_command_match": True,
        "resume_require_config_hash_match": False,
        "resume_require_git_commit_match": False,
        "force_rerun_all": False,
    }


def test_passed_server_validation_check_is_skipped(tmp_path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    record_path = logs_dir / "already_passed.json"
    command = ["python", "scripts/does_not_exist.py", "config/server_validation.yaml"]
    record_path.write_text(
        json.dumps(
            {
                "name": "already_passed",
                "command": command,
                "required_for_training": True,
                "status": "PASS",
                "returncode": 0,
            }
        ),
        encoding="utf-8",
    )

    result = _run_check(
        {
            "name": "already_passed",
            "command": command,
            "required_for_training": True,
        },
        output_root=tmp_path,
        validation_settings=_resume_settings(),
        git_commit="abc123",
    )

    assert result["status"] == "PASS"
    assert result["skipped"] is True
    assert result["resume_source"] == str(record_path)


def test_force_rerun_ignores_passed_record(tmp_path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    command = ["python", "-c", "print('rerun ok')"]
    (logs_dir / "rerunnable.json").write_text(
        json.dumps(
            {
                "name": "rerunnable",
                "command": command,
                "required_for_training": True,
                "status": "PASS",
                "returncode": 0,
            }
        ),
        encoding="utf-8",
    )
    settings = _resume_settings()
    settings["force_rerun_all"] = True

    result = _run_check(
        {
            "name": "rerunnable",
            "command": command,
            "required_for_training": True,
        },
        output_root=tmp_path,
        validation_settings=settings,
        git_commit="abc123",
    )

    assert result["status"] == "PASS"
    assert result["skipped"] is False
    assert "rerun ok" in result["stdout_tail"]
