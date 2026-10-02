"""Tests for remote runner Harbor dispatch helpers."""

from __future__ import annotations

from playground.remote_runner.dispatch import (
    discover_harbor_job_artifacts,
    filter_remote_harbor_payload_env,
    resolve_job_dir,
    run_harbor_job,
    write_incoming_job_config,
)


def test_write_incoming_job_config(tmp_path) -> None:
    path = write_incoming_job_config(
        repo_root=tmp_path,
        job_name="demo-job",
        config_yaml="job_name: demo-job\n",
    )
    assert path.is_file()
    assert "demo-job" in path.name


def test_run_harbor_job_invokes_command_runner(tmp_path, monkeypatch) -> None:
    calls: list[dict] = []

    def _fake_runner(command, *, cwd, env):
        calls.append({"command": list(command), "cwd": cwd, "env": env})
        return 0

    monkeypatch.setenv("REMOTE_RUNNER_HARBOR_COMMAND", "echo harbor")

    result = run_harbor_job(
        {
            "jobName": "demo-job",
            "configYaml": "job_name: demo-job\n",
            "repoRoot": str(tmp_path),
            "jobsDir": "jobs",
            "env": {"MATRIX_SURVEY_TASK_PATH": "application/tasks/survey_product-attitudes"},
        },
        command_runner=_fake_runner,
    )

    assert result["exitCode"] == 0
    assert calls
    assert calls[0]["command"][:2] == ["echo", "harbor"]
    assert "--yes" in calls[0]["command"]
    assert calls[0]["env"]["MATRIX_SURVEY_TASK_PATH"] == "application/tasks/survey_product-attitudes"
    pythonpath = calls[0]["env"]["PYTHONPATH"].split(":")
    assert str(tmp_path) in pythonpath
    assert str(tmp_path / "src") in pythonpath
    assert str(tmp_path / "environment" / "runtime") in pythonpath
    assert str(tmp_path / "environment" / "agents") in pythonpath
    assert str(tmp_path / "packages" / "playground" / "src") in pythonpath
    assert str(tmp_path / "application" / "playground") in pythonpath


def test_filter_remote_harbor_payload_env_drops_secrets() -> None:
    filtered = filter_remote_harbor_payload_env(
        {
            "PYTHONPATH": "/repo",
            "MATRIX_SURVEY_TASK_PATH": "application/tasks/demo",
            "ANTHROPIC_API_KEY": "sk-ant-secret",
            "HOME": "/Users/demo",
        }
    )
    assert filtered == {
        "PYTHONPATH": "/repo",
        "MATRIX_SURVEY_TASK_PATH": "application/tasks/demo",
    }


def test_run_harbor_job_ignores_secret_payload_env(tmp_path, monkeypatch) -> None:
    calls: list[dict] = []

    def _fake_runner(command, *, cwd, env):
        calls.append({"command": list(command), "cwd": cwd, "env": env})
        return 0

    monkeypatch.setenv("REMOTE_RUNNER_HARBOR_COMMAND", "echo harbor")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "worker-local-key")

    run_harbor_job(
        {
            "jobName": "demo-job",
            "configYaml": "job_name: demo-job\n",
            "repoRoot": str(tmp_path),
            "jobsDir": "jobs",
            "env": {
                "MATRIX_SURVEY_TASK_PATH": "application/tasks/survey_product-attitudes",
                "ANTHROPIC_API_KEY": "attacker-injected-key",
            },
        },
        command_runner=_fake_runner,
    )

    assert calls[0]["env"]["ANTHROPIC_API_KEY"] == "worker-local-key"
    assert calls[0]["env"]["MATRIX_SURVEY_TASK_PATH"] == "application/tasks/survey_product-attitudes"


def test_resolve_job_dir(tmp_path) -> None:
    repo = tmp_path / "repo"
    payload = {
        "jobName": "test-job",
        "jobsDir": "custom_jobs",
        "repoRoot": str(repo),
    }
    job_dir = resolve_job_dir(payload)
    assert job_dir == repo / "custom_jobs" / "test-job"

    # Explicit jobDir
    payload_explicit = {"jobDir": str(tmp_path / "explicit_dir")}
    assert resolve_job_dir(payload_explicit) == tmp_path / "explicit_dir"


def test_discover_harbor_job_artifacts_single_trial(tmp_path) -> None:
    import json
    repo = tmp_path
    job_dir = repo / "jobs" / "survey-1"
    out_dir = job_dir / "trial_a" / "artifacts" / "app" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    survey_data = {"score": 42}
    (out_dir / "survey_result.json").write_text(json.dumps(survey_data), encoding="utf-8")

    payload = {
        "jobName": "survey-1",
        "repoRoot": str(repo),
        "jobsDir": "jobs",
    }
    artifacts = discover_harbor_job_artifacts(payload)

    assert artifacts["survey_result.json"] == survey_data
    assert artifacts["trial_a/survey_result.json"] == survey_data
    assert artifacts["trial_a/artifacts/app/output/survey_result.json"] == survey_data
    assert artifacts["artifacts/app/output/survey_result.json"] == survey_data

    manifest = artifacts["artifact_manifest.json"]
    assert manifest["jobName"] == "survey-1"
    assert "trial_a/artifacts/app/output/survey_result.json" in manifest["artifacts"]
    assert manifest["trials"] == {"trial_a": ["trial_a/artifacts/app/output/survey_result.json"]}


def test_discover_harbor_job_artifacts_multi_trial(tmp_path) -> None:
    import json
    repo = tmp_path
    job_dir = repo / "jobs" / "survey-multi"
    t1_out = job_dir / "trial_1" / "artifacts" / "app" / "output"
    t2_out = job_dir / "trial_2" / "artifacts" / "app" / "output"
    t1_out.mkdir(parents=True, exist_ok=True)
    t2_out.mkdir(parents=True, exist_ok=True)

    (t1_out / "survey_result.json").write_text(json.dumps({"t": 1}), encoding="utf-8")
    (t2_out / "survey_result.json").write_text(json.dumps({"t": 2}), encoding="utf-8")
    (t1_out / "unique_report.json").write_text(json.dumps({"unique": True}), encoding="utf-8")

    payload = {
        "jobName": "survey-multi",
        "repoRoot": str(repo),
        "jobsDir": "jobs",
    }
    artifacts = discover_harbor_job_artifacts(payload)

    # survey_result.json is not unique across trials, so no basename shortcut
    assert "survey_result.json" not in artifacts

    # unique_report.json appears only once, so basename shortcut is present
    assert artifacts["unique_report.json"] == {"unique": True}

    # Deterministic relative paths are present
    assert artifacts["trial_1/artifacts/app/output/survey_result.json"] == {"t": 1}
    assert artifacts["trial_2/artifacts/app/output/survey_result.json"] == {"t": 2}
    assert artifacts["trial_1/survey_result.json"] == {"t": 1}
    assert artifacts["trial_2/survey_result.json"] == {"t": 2}

    manifest = artifacts["artifact_manifest.json"]
    assert len(manifest["artifacts"]) == 3
    assert set(manifest["trials"].keys()) == {"trial_1", "trial_2"}


def test_discover_harbor_job_artifacts_missing_dir(tmp_path) -> None:
    payload = {
        "jobName": "nonexistent",
        "repoRoot": str(tmp_path),
        "jobsDir": "jobs",
    }
    artifacts = discover_harbor_job_artifacts(payload)
    assert artifacts == {
        "artifact_manifest.json": {
            "jobName": "nonexistent",
            "artifacts": [],
            "entries": [],
            "trials": {},
        }
    }

