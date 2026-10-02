"""Tests for the general remote runner HTTP service."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from playground.remote_runner.dispatch import run_harbor_job
from playground.remote_runner.server import create_app


def test_remote_runner_server_harbor_job(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def _fake_harbor(payload, *, command_runner=None, harbor_command=None):
        del harbor_command

        def _runner(command, *, cwd, env):
            calls.append(list(command))
            return 0

        return run_harbor_job(payload, command_runner=_runner)

    from playground import remote_runner

    monkeypatch.setattr(remote_runner.server, "run_harbor_job", _fake_harbor)
    monkeypatch.setenv("REMOTE_RUNNER_INLINE", "1")

    client = TestClient(create_app(runs_dir=tmp_path / "runs"))
    response = client.post(
        "/v1/runs",
        json={
            "taskType": "harbor_job",
            "payload": {
                "jobName": "demo",
                "configYaml": "job_name: demo\n",
                "repoRoot": str(tmp_path),
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "succeeded"

    run_id = body["id"]
    artifact = client.get("/v1/runs/{}/artifacts/harbor_job_result.json".format(run_id))
    assert artifact.status_code == 200
    assert artifact.json()["jobName"] == "demo"
    assert calls


def test_remote_runner_server_rejects_unknown_task_type(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("REMOTE_RUNNER_INLINE", "1")
    client = TestClient(create_app(runs_dir=tmp_path / "runs"))
    response = client.post(
        "/v1/runs",
        json={"taskType": "survey", "payload": {}},
    )
    assert response.status_code == 200
    run_id = response.json()["id"]
    detail = client.get("/v1/runs/{}".format(run_id)).json()
    assert detail["status"] == "failed"


def test_remote_runner_server_harbor_job_single_trial_artifacts(tmp_path, monkeypatch) -> None:
    survey_data = {"answers": [{"questionId": "q1", "value": "Strongly agree"}]}

    def _fake_harbor(payload, *, command_runner=None, harbor_command=None):
        del harbor_command
        job_dir = tmp_path / "jobs" / "single-survey"
        out_dir = job_dir / "trial_001" / "artifacts" / "app" / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        import json
        (out_dir / "survey_result.json").write_text(json.dumps(survey_data), encoding="utf-8")

        def _runner(command, *, cwd, env):
            return 0

        return run_harbor_job(payload, command_runner=_runner)

    from playground import remote_runner

    monkeypatch.setattr(remote_runner.server, "run_harbor_job", _fake_harbor)
    monkeypatch.setenv("REMOTE_RUNNER_INLINE", "1")

    client = TestClient(create_app(runs_dir=tmp_path / "runs"))
    response = client.post(
        "/v1/runs",
        json={
            "taskType": "harbor_job",
            "payload": {
                "jobName": "single-survey",
                "configYaml": "job_name: single-survey\n",
                "repoRoot": str(tmp_path),
            },
        },
    )
    assert response.status_code == 200
    run_id = response.json()["id"]

    # Basename shortcut exposed because unique
    shortcut = client.get("/v1/runs/{}/artifacts/survey_result.json".format(run_id))
    assert shortcut.status_code == 200
    assert shortcut.json() == survey_data

    # Trial-scoped shortcut
    trial_shortcut = client.get("/v1/runs/{}/artifacts/trial_001/survey_result.json".format(run_id))
    assert trial_shortcut.status_code == 200
    assert trial_shortcut.json() == survey_data

    # Canonical relative path
    canonical = client.get("/v1/runs/{}/artifacts/trial_001/artifacts/app/output/survey_result.json".format(run_id))
    assert canonical.status_code == 200
    assert canonical.json() == survey_data

    # Manifest
    manifest_resp = client.get("/v1/runs/{}/artifacts/artifact_manifest.json".format(run_id))
    assert manifest_resp.status_code == 200
    manifest = manifest_resp.json()
    assert manifest["jobName"] == "single-survey"
    assert "trial_001/artifacts/app/output/survey_result.json" in manifest["artifacts"]
    assert "trial_001" in manifest["trials"]


def test_remote_runner_server_harbor_job_multi_trial_artifacts_uniqueness(tmp_path, monkeypatch) -> None:
    t1_data = {"trial": 1, "answers": [{"questionId": "q1", "value": "A"}]}
    t2_data = {"trial": 2, "answers": [{"questionId": "q1", "value": "B"}]}
    unique_metric = {"accuracy": 0.95}

    def _fake_harbor(payload, *, command_runner=None, harbor_command=None):
        del harbor_command
        import json
        job_dir = tmp_path / "jobs" / "multi-survey"
        out_dir_1 = job_dir / "trial_001" / "artifacts" / "app" / "output"
        out_dir_2 = job_dir / "trial_002" / "artifacts" / "app" / "output"
        out_dir_1.mkdir(parents=True, exist_ok=True)
        out_dir_2.mkdir(parents=True, exist_ok=True)
        (out_dir_1 / "survey_result.json").write_text(json.dumps(t1_data), encoding="utf-8")
        (out_dir_2 / "survey_result.json").write_text(json.dumps(t2_data), encoding="utf-8")
        (out_dir_2 / "unique_metric.json").write_text(json.dumps(unique_metric), encoding="utf-8")

        def _runner(command, *, cwd, env):
            return 0

        return run_harbor_job(payload, command_runner=_runner)

    from playground import remote_runner

    monkeypatch.setattr(remote_runner.server, "run_harbor_job", _fake_harbor)
    monkeypatch.setenv("REMOTE_RUNNER_INLINE", "1")

    client = TestClient(create_app(runs_dir=tmp_path / "runs"))
    response = client.post(
        "/v1/runs",
        json={
            "taskType": "harbor_job",
            "payload": {
                "jobName": "multi-survey",
                "configYaml": "job_name: multi-survey\n",
                "repoRoot": str(tmp_path),
            },
        },
    )
    assert response.status_code == 200
    run_id = response.json()["id"]

    # Duplicate basename is NOT exposed as a root shortcut
    ambiguous = client.get("/v1/runs/{}/artifacts/survey_result.json".format(run_id))
    assert ambiguous.status_code == 404

    # Unique basename IS exposed as a root shortcut
    unique_resp = client.get("/v1/runs/{}/artifacts/unique_metric.json".format(run_id))
    assert unique_resp.status_code == 200
    assert unique_resp.json() == unique_metric

    # Scoped paths work deterministically
    assert client.get("/v1/runs/{}/artifacts/trial_001/survey_result.json".format(run_id)).json() == t1_data
    assert client.get("/v1/runs/{}/artifacts/trial_002/survey_result.json".format(run_id)).json() == t2_data

    # Manifest lists both
    manifest = client.get("/v1/runs/{}/artifacts/artifact_manifest.json".format(run_id)).json()
    assert "trial_001/artifacts/app/output/survey_result.json" in manifest["artifacts"]
    assert "trial_002/artifacts/app/output/survey_result.json" in manifest["artifacts"]
    assert len(manifest["trials"]) == 2

