"""Unit tests for Harbor job launch config generation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from backend.service.harbor_job_service import (
    EXTERNAL_RUN_STALE_AFTER_SEC,
    HarborJobService,
    HarborLaunchRecord,
    _job_list_status,
)


def test_launch_writes_job_config(tmp_path, monkeypatch):
    repo = tmp_path
    jobs_dir = repo / "jobs"
    jobs_dir.mkdir()
    (repo / "persona" / "datasets" / "matraix-persona-dev-sample").mkdir(parents=True)
    (repo / "persona" / "datasets" / "matraix-persona-dev-sample" / "persona_0001.yaml").write_text(
        "persona_id: '0001'\nversion: '1.0'\nsource: Nemotron\ndimensions: {}\n",
        encoding="utf-8",
    )
    (repo / "persona" / "datasets" / "matraix-persona-dev-sample" / "persona_0002.yaml").write_text(
        "persona_id: '0002'\nversion: '1.0'\nsource: OASIS\ndimensions: {}\n",
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    def _fake_run(command, *, cwd, env):
        calls.append({"command": list(command), "cwd": cwd, "env": dict(env)})
        return 0

    monkeypatch.setattr(
        "playground.harbor.playground._repo_root",
        lambda: repo,
    )
    service = HarborJobService(
        repo_root=repo,
        jobs_dir=jobs_dir,
        generated_configs_dir=repo / "configs" / "jobs" / "application-task-job-recipe",
        command_runner=_fake_run,
        harbor_command=("echo", "harbor"),
    )
    service._executor = _FakeExecutor()

    job_name = service.launch(
        task_path="application/tasks/example-survey_product-feedback",
        sample_size=2,
        seed=1,
        persona_pool="persona/datasets/matraix-persona-dev-sample",
        persona_model="anthropic/claude-haiku-4-5",
        job_name="test-harbor-job",
    )
    assert job_name == "test-harbor-job"
    config_path = repo / "configs" / "jobs" / "application-task-job-recipe" / "test-harbor-job.yaml"
    assert config_path.is_file()
    text = config_path.read_text(encoding="utf-8")
    assert "persona_0001.yaml" in text or "persona_0002.yaml" in text
    legacy_config = yaml.safe_load(text)
    assert legacy_config["tasks"] == [
        {"path": "application/tasks/example-survey_product-feedback"}
    ]
    assert "trial_plan" not in legacy_config
    assert service._executor.calls
    fn, args, kwargs = service._executor.calls[0]
    assert fn.__name__ == "_run_local_distributed"
    fn(*args, **kwargs)
    assert calls
    assert calls[0]["command"][0:4] == ["echo", "harbor", "trials", "start"]
    env = calls[0]["env"]
    assert isinstance(env, dict)
    assert env["MATRIX_SURVEY_TASK_PATH"] == "application/tasks/example-survey_product-feedback"

    detail = service.get_job("test-harbor-job")
    assert detail is not None
    assert detail["launch"]["status"] == "completed"
    assert len(detail["trials"]) == 2

    service.shutdown()


def test_launch_preserves_compiled_trial_plan(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=jobs_dir,
        generated_configs_dir=tmp_path / "configs",
        command_runner=lambda *args, **kwargs: 0,
        harbor_command=("echo", "harbor"),
    )
    service._executor = _FakeExecutor()
    compiled = {
        "job_name": "compiled-job",
        "jobs_dir": "jobs",
        "n_attempts": 2,
        "n_concurrent_trials": 3,
        "tasks": [
            {"path": "application/tasks/survey_a"},
            {"path": "application/tasks/survey_b"},
        ],
        "agents": [
            {
                "name": "persona-json-survey",
                "kwargs": {
                    "persona_path": "persona_0001.yaml",
                    "survey_task_path": "application/tasks/survey_a",
                },
            },
            {
                "name": "persona-json-survey",
                "kwargs": {
                    "persona_path": "persona_0002.yaml",
                    "survey_task_path": "application/tasks/survey_b",
                },
            },
        ],
        "trial_plan": [
            {"task_index": 0, "agent_index": 0},
            {"task_index": 1, "agent_index": 1},
        ],
        "_job_meta": {"condition_assignment": {"seed": 7}},
    }

    def _unexpected_build(*args, **kwargs):
        raise AssertionError("compiled config must not be rebuilt")

    monkeypatch.setattr(
        "backend.service.harbor_job_service.build_application_job_config",
        _unexpected_build,
    )
    job_name = service.launch(
        task_path="application/tasks/survey_a",
        compiled_job_config=compiled,
        cohort_id="not-a-local-cohort",
        sample_size=0,
        execution_plane="harbor",
    )

    assert job_name == "compiled-job"
    config = yaml.safe_load((tmp_path / "configs" / "compiled-job.yaml").read_text())
    for key in ("tasks", "agents", "trial_plan", "n_attempts", "n_concurrent_trials"):
        assert config[key] == compiled[key]
    assert compiled["_job_meta"] == {"condition_assignment": {"seed": 7}}
    assert "_job_meta" not in config
    launch_meta = service._read_json(service._launch_meta_path(job_name))
    assert launch_meta["jobMeta"] == compiled["_job_meta"]
    assert service._executor.calls[0][0].__name__ == "_run_local_distributed"
    service.shutdown()


def test_launch_normalizes_compiled_trial_plan_indices_without_mutating_input(tmp_path):
    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=tmp_path / "jobs",
        generated_configs_dir=tmp_path / "configs",
    )
    service._executor = _FakeExecutor()
    compiled = {
        "job_name": "normalized-plan",
        "tasks": [{"path": "application/tasks/survey_a"}],
        "agents": [{"name": "persona-json-survey"}],
        "trial_plan": [{"task_index": "0", "agent_index": "0"}],
    }

    service.launch(
        task_path="application/tasks/survey_a",
        compiled_job_config=compiled,
        execution_plane="harbor",
    )

    emitted = yaml.safe_load((tmp_path / "configs" / "normalized-plan.yaml").read_text())
    assert emitted["trial_plan"] == [{"task_index": 0, "agent_index": 0}]
    assert compiled["trial_plan"] == [{"task_index": "0", "agent_index": "0"}]
    service.shutdown()


@pytest.mark.parametrize(
    ("trial_plan", "message"),
    [
        ([{"task_index": 1, "agent_index": 0}], "task_index"),
        ([{"task_index": 0, "agent_index": 1}], "agent_index"),
    ],
)
def test_launch_rejects_invalid_compiled_trial_plan(
    tmp_path, trial_plan, message
):
    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=tmp_path / "jobs",
        generated_configs_dir=tmp_path / "configs",
    )
    with pytest.raises(ValueError, match=message):
        service.launch(
            task_path="application/tasks/survey_a",
            compiled_job_config={
                "tasks": [{"path": "application/tasks/survey_a"}],
                "agents": [{"name": "persona-json-survey"}],
                "trial_plan": trial_plan,
            },
        )
    service.shutdown()


@pytest.mark.parametrize(
    ("family", "dispatch_name"),
    [("modal", "_dispatch_modal_host"), ("gcp", "_dispatch_gke_host")],
)
def test_launch_cloud_shards_preserve_compiled_pairings(
    tmp_path, monkeypatch, family, dispatch_name
):
    monkeypatch.setenv("MATRIX_HOST_PACK_CONCURRENCY", "1")
    monkeypatch.setenv("MATRIX_SHARD_CONCURRENCY", "2")
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=jobs_dir,
        generated_configs_dir=tmp_path / "configs",
        modal_host_job_runner=object(),
        gke_host_job_runner=object(),
    )
    service._executor = _FakeExecutor()
    compiled = {
        "job_name": "compiled-cloud",
        "jobs_dir": "jobs",
        "n_concurrent_trials": 2,
        "environment": {"type": "host", "env": {"KEEP_ME": "yes"}},
        "tasks": [
            {"path": "application/tasks/survey_a"},
            {"path": "application/tasks/survey_b"},
        ],
        "agents": [
            {"name": "persona-json-survey", "kwargs": {"persona_path": "persona_0001.yaml"}},
            {"name": "persona-json-survey", "kwargs": {"persona_path": "persona_0002.yaml"}},
        ],
        "trial_plan": [
            {"task_index": 0, "agent_index": 0},
            {"task_index": 1, "agent_index": 1},
        ],
    }
    service.launch(
        task_path="application/tasks/survey_a",
        compiled_job_config=compiled,
        compute_family=family,
        execution_plane="harbor",
    )

    parent = yaml.safe_load((tmp_path / "configs" / "compiled-cloud.yaml").read_text())
    assert parent["tasks"] == compiled["tasks"]
    assert parent["agents"] == compiled["agents"]
    assert parent["trial_plan"] == compiled["trial_plan"]
    assert parent["environment"] == compiled["environment"]
    shard_paths = sorted((tmp_path / "configs").glob("compiled-cloud.shard-*.yaml"))
    assert len(shard_paths) == 2
    pairings = set()
    for path in shard_paths:
        shard = yaml.safe_load(path.read_text())
        assert shard["tasks"] == compiled["tasks"]
        assert len(shard["agents"]) == len(shard["trial_plan"]) == 1
        entry = shard["trial_plan"][0]
        pairings.add(
            (
                shard["tasks"][entry["task_index"]]["path"],
                shard["agents"][entry["agent_index"]]["kwargs"]["persona_path"],
            )
        )
    assert pairings == {
        ("application/tasks/survey_a", "persona_0001.yaml"),
        ("application/tasks/survey_b", "persona_0002.yaml"),
    }
    parent_path = tmp_path / "configs" / "compiled-cloud.yaml"
    assert len(service._cloud_shard_work_items("compiled-cloud", parent_path)) == 2
    shard_paths[0].unlink()
    with pytest.raises(ValueError, match="planned shards"):
        service._cloud_shard_work_items("compiled-cloud", parent_path)
    shard_paths[1].unlink()
    with pytest.raises(ValueError, match="planned shards"):
        service._cloud_shard_work_items("compiled-cloud", parent_path)
    assert service._executor.calls[0][0].__name__ == dispatch_name
    service.shutdown()


def test_launch_rejects_compiled_jobs_dir_mismatch(tmp_path):
    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=tmp_path / "jobs",
        generated_configs_dir=tmp_path / "configs",
    )
    with pytest.raises(ValueError, match="jobs_dir"):
        service.launch(
            task_path="application/tasks/survey_a",
            compiled_job_config={
                "jobs_dir": "somewhere-else",
                "tasks": [{"path": "application/tasks/survey_a"}],
                "agents": [{"name": "persona-json-survey"}],
            },
        )
    service.shutdown()


def test_launch_with_frozen_cohort(tmp_path, monkeypatch):
    from application.playground.backend.tests.test_persona_pool_service import _write_pool

    repo = tmp_path
    _write_pool(repo)
    jobs_dir = repo / "jobs"
    jobs_dir.mkdir()

    from backend.service.persona_pool_service import PersonaPoolService

    pool_service = PersonaPoolService(repo_root=repo)
    pool_service.save_cohort(
        cohort_id="frozen-oasis",
        kind="frozen",
        seed=1,
        sample_size=1,
        dimension_filters={"economic_motivation": "Indifferent"},
    )

    calls: list[list[str]] = []

    def _fake_run(command, *, cwd, env):
        calls.append(list(command))
        return 0

    monkeypatch.setattr(
        "playground.harbor.playground._repo_root",
        lambda: repo,
    )
    service = HarborJobService(
        repo_root=repo,
        jobs_dir=jobs_dir,
        generated_configs_dir=repo / "configs" / "jobs" / "application-task-job-recipe",
        command_runner=_fake_run,
        harbor_command=("echo", "harbor"),
    )
    service._executor = _FakeExecutor()

    job_name = service.launch(
        task_path="application/tasks/example-survey_product-feedback",
        cohort_id="frozen-oasis",
        persona_model="anthropic/claude-haiku-4-5",
        job_name="cohort-job",
    )
    assert job_name == "cohort-job"
    config_path = repo / "configs" / "jobs" / "application-task-job-recipe" / "cohort-job.yaml"
    text = config_path.read_text(encoding="utf-8")
    assert "persona_0002.yaml" in text
    assert "# Cohort: frozen-oasis" in text
    assert service._executor.calls
    assert service._executor.calls[0][0].__name__ == "_run_local_distributed"
    service.shutdown()


def test_launch_with_explicit_persona_ids(tmp_path, monkeypatch):
    repo = tmp_path
    jobs_dir = repo / "jobs"
    jobs_dir.mkdir()
    pool = repo / "persona" / "datasets" / "matraix-persona-dev-sample"
    pool.mkdir(parents=True)
    (pool / "persona_0042.yaml").write_text(
        "persona_id: '0042'\nversion: '1.0'\nsource: Nemotron\ndimensions: {}\n",
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def _fake_run(command, *, cwd, env):
        calls.append(list(command))
        return 0

    monkeypatch.setattr(
        "playground.harbor.playground._repo_root",
        lambda: repo,
    )
    service = HarborJobService(
        repo_root=repo,
        jobs_dir=jobs_dir,
        generated_configs_dir=repo / "configs" / "jobs" / "application-task-job-recipe",
        command_runner=_fake_run,
        harbor_command=("echo", "harbor"),
    )
    service._executor = _FakeExecutor()

    job_name = service.launch(
        task_path="application/tasks/example-survey_product-feedback",
        persona_ids=["0042"],
        persona_model="anthropic/claude-haiku-4-5",
        execution_mode="auto",
        job_name="explicit-persona-job",
    )
    assert job_name == "explicit-persona-job"
    config_path = repo / "configs" / "jobs" / "application-task-job-recipe" / "explicit-persona-job.yaml"
    text = config_path.read_text(encoding="utf-8")
    assert "persona_0042.yaml" in text
    assert "# Mode: auto" in text
    assert "# Trial profile: json_survey" in text
    assert "type: host" in text
    assert "application/tasks/example-survey_product-feedback" in text
    assert service._executor.calls
    assert service._executor.calls[0][0].__name__ == "_run_local_distributed"
    service.shutdown()


def test_retry_failed_reruns_only_failed_trials(tmp_path, monkeypatch):
    repo = tmp_path
    jobs_dir = repo / "jobs"
    jobs_dir.mkdir()
    pool = repo / "persona" / "datasets" / "matraix-persona-dev-sample"
    pool.mkdir(parents=True)
    (pool / "persona_0001.yaml").write_text(
        "persona_id: '0001'\nversion: '1.0'\nsource: Nemotron\ndimensions: {}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("playground.harbor.playground._repo_root", lambda: repo)
    service = HarborJobService(
        repo_root=repo,
        jobs_dir=jobs_dir,
        generated_configs_dir=repo / "configs" / "jobs" / "application-task-job-recipe",
        command_runner=lambda *args, **kwargs: 0,
        harbor_command=("echo", "harbor"),
    )
    service._executor = _FakeExecutor()

    job_name = service.launch(
        task_path="application/tasks/example-survey_product-feedback",
        persona_ids=["0001"],
        persona_model="anthropic/claude-haiku-4-5",
        job_name="retry-job",
    )
    # Launch metadata is persisted so a retry can replay the identical dispatch.
    meta_path = service._launch_meta_path(job_name)
    assert meta_path.is_file()
    assert len(service._executor.calls) == 1

    # Mark the launch record finished so retry is allowed.
    service._launches[job_name].status = "completed"

    job_dir = jobs_dir / job_name
    ok_dir = job_dir / "trial-ok"
    ok_dir.mkdir(parents=True)
    (ok_dir / "result.json").write_text("{}", encoding="utf-8")
    fail_dir = job_dir / "trial-fail"
    fail_dir.mkdir(parents=True)
    (fail_dir / "result.json").write_text(
        json.dumps({"exception_info": {"exception_message": "boom"}}),
        encoding="utf-8",
    )
    (job_dir / "result.json").write_text(
        json.dumps({"finished_at": "2026-07-01T12:00:00Z", "stats": {"n_errored_trials": 1}}),
        encoding="utf-8",
    )

    assert service.failed_trial_count(job_name) == 1

    result = service.retry_failed(job_name)
    assert result == {"jobName": job_name, "retried": 1}

    # Succeeded trial preserved; failed trial removed for Harbor to re-run.
    assert ok_dir.is_dir()
    assert not fail_dir.exists()
    # Job-level completion cleared so status/reporting recompute.
    assert not (job_dir / "result.json").exists()
    # A fresh dispatch was submitted and the record is active again.
    assert len(service._executor.calls) == 2
    assert service._launches[job_name].status == "queued"

    service.shutdown()


def test_planned_cloud_retry_uses_only_failed_pairs_and_attempts(tmp_path, monkeypatch):
    from backend.service.modal_host_job import ModalHostJobResult, save_cloud_run
    from harbor.models.trial.config import AgentConfig, TaskConfig

    job_name = "planned-cloud-retry"
    job_dir = tmp_path / "jobs" / job_name
    job_dir.mkdir(parents=True)
    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=tmp_path / "jobs",
        generated_configs_dir=tmp_path / "configs",
        modal_host_job_runner=object(),
    )
    service._executor = _FakeExecutor()
    config = {
        "job_name": job_name,
        "jobs_dir": "jobs",
        "n_attempts": 2,
        "n_concurrent_trials": 4,
        "tasks": [{"path": "task-a"}, {"path": "task-b"}],
        "agents": [
            {"name": "persona-json-survey", "kwargs": {"persona_path": "persona_0001.yaml"}},
            {"name": "persona-json-survey", "kwargs": {"persona_path": "persona_0002.yaml"}},
        ],
        "trial_plan": [
            {"task_index": 0, "agent_index": 0},
            {"task_index": 1, "agent_index": 1},
            {"task_index": 0, "agent_index": 0},
        ],
    }
    config_path = tmp_path / "configs" / f"{job_name}.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    service._persist_launch_meta(
        job_name,
        {
            "configPath": str(config_path.relative_to(tmp_path)),
            "jobConfig": config,
            "computeDispatch": "modal_jobs",
            "computeFamily": "modal",
            "executionPlane": "harbor",
            "shards": [
                {"key": "s0", "index": 0, "kind": "cloud", "personaIds": ["0001", "0002"]}
            ],
        },
    )
    save_cloud_run(
        job_dir,
        {
            "jobName": job_name,
            "dispatch": "modal_jobs",
            "status": "completed",
            "shards": [{"key": "s0", "callId": "old-call", "state": "completed"}],
        },
    )
    for trial_name, task_index, agent_index, failed in [
        ("success-a-1", 0, 0, False),
        ("success-a-2", 0, 0, False),
        ("success-a-3", 0, 0, False),
        ("failed-a", 0, 0, True),
        ("success-b", 1, 1, False),
        ("failed-b", 1, 1, True),
    ]:
        trial_dir = job_dir / trial_name
        trial_dir.mkdir()
        (trial_dir / "config.json").write_text(
            json.dumps({"task": config["tasks"][task_index], "agent": config["agents"][agent_index]}),
            encoding="utf-8",
        )
        (trial_dir / "result.json").write_text(
            json.dumps({"exception_info": {"exception_type": "TimeoutError"}} if failed else {}),
            encoding="utf-8",
        )

    assert service.retry_failed(job_name) == {"jobName": job_name, "retried": 2}
    assert (job_dir / "success-a-1" / "result.json").is_file()
    assert (job_dir / "success-a-2" / "result.json").is_file()
    assert (job_dir / "success-a-3" / "result.json").is_file()
    assert (job_dir / "success-b" / "result.json").is_file()
    assert not (job_dir / "failed-a").exists()
    assert not (job_dir / "failed-b").exists()
    meta = service._read_json(service._launch_meta_path(job_name))
    assert meta is not None
    retry_row = meta["retryShards"][0]
    assert retry_row["key"].startswith("retry-")
    assert retry_row["key"] != "s0"
    retry_cfg = yaml.safe_load((tmp_path / retry_row["configPath"]).read_text())
    assert retry_cfg["n_attempts"] == 1
    assert retry_cfg["tasks"] == [
        TaskConfig.model_validate(task).model_dump(mode="json")
        for task in config["tasks"]
    ]
    assert retry_cfg["agents"] == [
        AgentConfig.model_validate(agent).model_dump(mode="json")
        for agent in config["agents"]
    ]
    assert retry_cfg["trial_plan"] == [
        {"task_index": 0, "agent_index": 0},
        {"task_index": 1, "agent_index": 1},
    ]
    assert service._executor.calls[0][0].__name__ == "_dispatch_modal_host"

    service._launches[job_name].status = "failed"
    service._maybe_rollup_job_result(job_name)
    assert service._read_json(service._launch_meta_path(job_name))["retryShards"][0] == retry_row
    for result in [{}, {"exception_info": {"exception_type": "TimeoutError"}}]:
        partial = job_dir / "partial-output"
        partial.mkdir()
        (partial / "config.json").write_text(
            json.dumps({"task": config["tasks"][0], "agent": config["agents"][0]}),
            encoding="utf-8",
        )
        (partial / "result.json").write_text(json.dumps(result), encoding="utf-8")
        with pytest.raises(ValueError, match="partial retry output"):
            service.retry_failed(job_name)
        assert (partial / "result.json").is_file()
        shutil.rmtree(partial)

    assert service.retry_failed(job_name) == {"jobName": job_name, "retried": 2}
    assert all((job_dir / name / "result.json").is_file() for name in [
        "success-a-1", "success-a-2", "success-a-3", "success-b"
    ])
    resumed_meta = service._read_json(service._launch_meta_path(job_name))
    resumed_row = resumed_meta["retryShards"][0]
    assert resumed_row["key"].startswith("retry-")
    assert resumed_row["key"] != retry_row["key"]
    resumed_cfg = yaml.safe_load((tmp_path / resumed_row["configPath"]).read_text())
    for key in ("tasks", "agents", "trial_plan", "n_attempts"):
        assert resumed_cfg[key] == retry_cfg[key]
    retry_row = resumed_row

    class _SpawnRunner:
        def __init__(self):
            self.spawned = []
            self.waited = []

        def spawn(self, request):
            self.spawned.append(request.shard_key)
            assert request.jobs_dir == f"jobs/{request.shard_key}"
            return "new-call", object()

        def wait(self, request, *, call, call_id):
            self.waited.append(call_id)
            return ModalHostJobResult(exit_code=0)

    runner = _SpawnRunner()
    monkeypatch.setattr(
        "backend.service.modal_host_job.materialize_modal_artifacts",
        lambda *args, **kwargs: None,
    )
    work = service._cloud_shard_work_items(job_name, config_path)
    codes, errors = service._spawn_and_wait_cloud_shards(
        "modal_jobs", job_name, work, {}, runner
    )
    assert (codes, errors) == ([0], [])
    assert runner.spawned == [retry_row["key"]]
    assert runner.waited == ["new-call"]
    service.shutdown()


def test_planned_cloud_retry_rejects_unplanned_failed_pair(tmp_path):
    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=tmp_path / "jobs",
        generated_configs_dir=tmp_path / "configs",
    )
    failed = tmp_path / "jobs" / "job" / "failed"
    failed.mkdir(parents=True)
    (failed / "config.json").write_text(
        json.dumps({"task": {"path": "other-task"}, "agent": {"name": "oracle"}}),
        encoding="utf-8",
    )
    meta = {
        "jobConfig": {
            "tasks": [{"path": "task-a"}],
            "agents": [{"name": "oracle"}],
            "trial_plan": [{"task_index": 0, "agent_index": 0}],
        }
    }

    with pytest.raises(ValueError, match="pairing is absent"):
        service._planned_retry_shard(meta, [failed], job_name="job", kind="cloud")
    service.shutdown()


@pytest.mark.parametrize(
    "case",
    ["empty", "non-list", "invalid-row", "missing-file", "partial", "tampered", "erased-plan"],
)
def test_planned_cloud_resume_rejects_invalid_retry_shards(tmp_path, case):
    job_name = "planned-cloud-resume"
    config_path = tmp_path / "configs" / f"{job_name}.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        yaml.safe_dump(
            {
                "job_name": job_name,
                "trial_plan": [{"task_index": 0, "agent_index": 0}],
            }
            if case == "erased-plan"
            else {"job_name": job_name}
        ),
        encoding="utf-8",
    )
    retry_yaml = "job_name: planned-cloud-resume\n"
    (config_path.parent / "existing.yaml").write_text(retry_yaml, encoding="utf-8")
    valid_row = {
        "key": "retry-one",
        "configPath": "configs/existing.yaml",
        "configSha256": hashlib.sha256(retry_yaml.encode("utf-8")).hexdigest(),
    }
    missing_row = {
        "key": "retry-two",
        "configPath": "configs/missing.yaml",
        "configSha256": "0" * 64,
    }
    retry_shards = {
        "empty": [],
        "non-list": "invalid",
        "invalid-row": [{}],
        "missing-file": [missing_row],
        "partial": [valid_row, missing_row],
        "tampered": [valid_row],
        "erased-plan": [],
    }[case]
    if case == "tampered":
        (config_path.parent / "existing.yaml").write_text(
            "job_name: altered\n", encoding="utf-8"
        )
    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=tmp_path / "jobs",
        generated_configs_dir=config_path.parent,
    )
    service._persist_launch_meta(
        job_name,
        {
            "jobConfig": {
                "trial_plan": [] if case == "erased-plan" else [{"task_index": 0, "agent_index": 0}]
            },
            "retryShards": retry_shards,
        },
    )

    with pytest.raises(ValueError, match="planned retry"):
        service._cloud_shard_work_items(job_name, config_path)
    service.shutdown()


def test_planned_cloud_retry_rejects_negative_saved_index_before_cleanup(tmp_path):
    job_name = "negative-plan-index"
    job_dir = tmp_path / "jobs" / job_name
    failed = job_dir / "failed-trial"
    failed.mkdir(parents=True)
    (failed / "result.json").write_text(
        json.dumps({"exception_info": {"exception_type": "TimeoutError"}}),
        encoding="utf-8",
    )
    (failed / "config.json").write_text(
        json.dumps({"task": {"path": "task-a"}, "agent": {"name": "oracle"}}),
        encoding="utf-8",
    )
    config_path = tmp_path / "configs" / f"{job_name}.yaml"
    config_path.parent.mkdir()
    job_config = {
        "tasks": [{"path": "task-a"}],
        "agents": [{"name": "oracle"}],
        "trial_plan": [{"task_index": -1, "agent_index": 0}],
    }
    config_path.write_text(yaml.safe_dump(job_config), encoding="utf-8")
    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=tmp_path / "jobs",
        generated_configs_dir=config_path.parent,
    )
    service._persist_launch_meta(
        job_name,
        {
            "configPath": str(config_path.relative_to(tmp_path)),
            "computeDispatch": "modal_jobs",
            "jobConfig": job_config,
        },
    )

    with pytest.raises(ValueError, match="task_index"):
        service.retry_failed(job_name)
    assert (failed / "result.json").is_file()
    service.shutdown()


def test_planned_retry_keeps_failure_when_work_items_cannot_be_persisted(
    tmp_path, monkeypatch
):
    job_name = "retry-metadata-write"
    failed = tmp_path / "jobs" / job_name / "failed-trial"
    failed.mkdir(parents=True)
    (failed / "result.json").write_text(
        json.dumps({"exception_info": {"exception_type": "TimeoutError"}}),
        encoding="utf-8",
    )
    config = {
        "tasks": [{"path": "task-a"}],
        "agents": [{"name": "oracle"}],
        "trial_plan": [{"task_index": 0, "agent_index": 0}],
    }
    (failed / "config.json").write_text(
        json.dumps({"task": config["tasks"][0], "agent": config["agents"][0]}),
        encoding="utf-8",
    )
    config_path = tmp_path / "configs" / f"{job_name}.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=tmp_path / "jobs",
        generated_configs_dir=config_path.parent,
    )
    service._persist_launch_meta(
        job_name,
        {
            "configPath": str(config_path.relative_to(tmp_path)),
            "computeDispatch": "modal_jobs",
            "jobConfig": config,
        },
    )
    monkeypatch.setattr(service, "_persist_launch_meta", lambda *_args: None)

    with pytest.raises(ValueError, match="retryShards could not be persisted"):
        service.retry_failed(job_name)
    assert (failed / "result.json").is_file()
    service.shutdown()


@pytest.mark.parametrize("stale", [None, "generated", "metadata", "condition-marker"])
def test_unsharded_planned_retry_checks_generated_config_before_cleanup(tmp_path, stale):
    job_name = "local-plan-retry"
    job_dir = tmp_path / "jobs" / job_name
    failed = job_dir / "failed-trial"
    failed.mkdir(parents=True)
    (failed / "result.json").write_text(
        json.dumps({"exception_info": {"exception_type": "TimeoutError"}}),
        encoding="utf-8",
    )
    config = {
        "job_name": job_name,
        "tasks": [{"path": "task-a"}, {"path": "task-b"}],
        "agents": [{"name": "oracle"}],
        "trial_plan": [{"task_index": 0, "agent_index": 0}],
    }
    config_path = tmp_path / "configs" / f"{job_name}.yaml"
    config_path.parent.mkdir()
    generated = (
        {**config, "trial_plan": [{"task_index": 1, "agent_index": 0}]}
        if stale == "generated"
        else {**config, "trial_plan": []}
        if stale == "condition-marker"
        else config
    )
    config_path.write_text(yaml.safe_dump(generated), encoding="utf-8")
    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=tmp_path / "jobs",
        generated_configs_dir=config_path.parent,
    )
    service._executor = _FakeExecutor()
    service._persist_launch_meta(
        job_name,
        {
            "configPath": str(config_path.relative_to(tmp_path)),
            "jobConfig": (
                {**config, "trial_plan": []}
                if stale in {"metadata", "condition-marker"}
                else config
            ),
            **(
                {"jobMeta": {"condition_assignment": {"seed": 7}}}
                if stale == "condition-marker"
                else {}
            ),
        },
    )

    if stale:
        with pytest.raises(ValueError, match="generated config changed since launch"):
            service.retry_failed(job_name)
        assert (failed / "result.json").is_file()
        assert not service._executor.calls
    else:
        assert service.retry_failed(job_name) == {"jobName": job_name, "retried": 1}
        assert not failed.exists()
        assert service._executor.calls
    service.shutdown()


@pytest.mark.parametrize("compute_dispatch", [None, "modal_jobs"])
def test_dataset_backed_planned_retry_is_unsupported_without_removing_failure(
    tmp_path, compute_dispatch
):
    job_name = "dataset-plan"
    job_dir = tmp_path / "jobs" / job_name
    failed = job_dir / "failed-trial"
    failed.mkdir(parents=True)
    (failed / "result.json").write_text(
        json.dumps({"exception_info": {"exception_type": "TimeoutError"}}),
        encoding="utf-8",
    )
    (failed / "config.json").write_text(
        json.dumps({"task": {"path": "resolved-task"}, "agent": {"name": "oracle"}}),
        encoding="utf-8",
    )
    config_path = tmp_path / "configs" / f"{job_name}.yaml"
    config_path.parent.mkdir()
    job_config = {
        "tasks": [],
        "datasets": [{"path": "dataset"}],
        "agents": [{"name": "oracle"}],
        "trial_plan": [{"task_index": 0, "agent_index": 0}],
    }
    config_path.write_text(yaml.safe_dump(job_config), encoding="utf-8")
    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=tmp_path / "jobs",
        generated_configs_dir=tmp_path / "configs",
    )
    service._persist_launch_meta(
        job_name,
        {
            "configPath": str(config_path.relative_to(tmp_path)),
            "computeDispatch": compute_dispatch,
            "jobConfig": job_config,
        },
    )

    with pytest.raises(ValueError, match="Dataset-backed trial_plan retry is unsupported"):
        service.retry_failed(job_name)

    assert failed.is_dir()
    assert (failed / "result.json").is_file()
    service.shutdown()


@pytest.mark.parametrize("planned", [False, True])
def test_cloud_dispatch_normalizes_only_planned_trial_config_for_resume(
    tmp_path, monkeypatch, planned
):
    from backend.service.modal_host_job import ModalHostJobResult
    from harbor.models.trial.config import TrialConfig

    job_name = "cloud-resume"
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / job_name
    job_dir.mkdir(parents=True)
    task = {"path": "task-a"}
    agent = {
        "name": "persona-json-survey",
        "kwargs": {"persona_path": "persona_0001.yaml"},
    }
    config = {
        "job_name": job_name,
        "jobs_dir": "jobs",
        "tasks": [task],
        "agents": [agent],
    }
    if planned:
        config["trial_plan"] = [{"task_index": 0, "agent_index": 0}]
    config_path = tmp_path / "configs" / f"{job_name}.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    class _Runner:
        calls = 0

        def run(self, request):
            self.calls += 1
            assert request.job_name == job_name
            return ModalHostJobResult(exit_code=0)

    runner = _Runner()
    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=jobs_dir,
        generated_configs_dir=config_path.parent,
        modal_host_job_runner=runner,
    )
    service._persist_launch_meta(
        job_name,
        {
            "jobConfig": config,
            "configPath": str(config_path.relative_to(tmp_path)),
            "computeDispatch": "modal_jobs",
        },
    )
    service._launches[job_name] = HarborLaunchRecord(job_name=job_name)

    remote_trials_dir = f"/workspace/jobs/s0/{job_name}"

    def _materialize(_result, *, jobs_dir, job_name):
        trial_dir = jobs_dir / job_name / "trial-a"
        trial_dir.mkdir(parents=True)
        (trial_dir / "config.json").write_text(
            json.dumps(
                {
                    "task": task,
                    "agent": agent,
                    "trials_dir": remote_trials_dir,
                }
            ),
            encoding="utf-8",
        )
        (trial_dir / "result.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "backend.service.modal_host_job.materialize_modal_artifacts", _materialize
    )
    monkeypatch.setattr(service, "_maybe_rollup_job_result", lambda _job_name: None)
    monkeypatch.setattr(service, "_maybe_run_host_verifier", lambda _job_name: None)
    monkeypatch.setattr(service, "_maybe_generate_post_run_feedback", lambda _job_name: None)
    monkeypatch.setattr(service, "_maybe_schedule_reporting", lambda *_args: None)

    service._dispatch_modal_host(job_name, config_path)

    assert runner.calls == 1
    assert service._launches[job_name].status == "completed"
    merged = TrialConfig.model_validate_json(
        (job_dir / "trial-a" / "config.json").read_text()
    )
    parent = TrialConfig.model_validate(
        {"task": task, "agent": agent, "trials_dir": str(Path("jobs") / job_name)}
    )
    if planned:
        assert merged == parent
    else:
        assert merged.trials_dir == Path(remote_trials_dir)
    service.shutdown()


def test_planned_harbor_fanout_retry_runs_only_failed_attempt_and_merges(
    tmp_path, monkeypatch
):
    job_name = "planned-harbor-retry"
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / job_name
    job_dir.mkdir(parents=True)
    commands = []

    def _run_retry(command, *, cwd, env):
        commands.append(list(command))
        retry_cfg = yaml.safe_load((cwd / command[command.index("-c") + 1]).read_text())
        worker_dir = cwd / retry_cfg["jobs_dir"] / job_name
        retried = worker_dir / "retried-a"
        retried.mkdir(parents=True)
        (retried / "config.json").write_text(
            json.dumps(
                {
                    "task": retry_cfg["tasks"][0],
                    "agent": retry_cfg["agents"][0],
                    "trials_dir": str(worker_dir),
                }
            ),
            encoding="utf-8",
        )
        (retried / "result.json").write_text("{}", encoding="utf-8")
        return 0

    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=jobs_dir,
        generated_configs_dir=tmp_path / "configs",
        command_runner=_run_retry,
        harbor_command=("echo", "harbor"),
    )
    service._executor = _FakeExecutor()
    config = {
        "job_name": job_name,
        "jobs_dir": "jobs",
        "n_attempts": 2,
        "n_concurrent_trials": 4,
        "tasks": [{"path": "task-a"}, {"path": "task-b"}],
        "agents": [
            {"name": "persona-json-survey", "kwargs": {"persona_path": "persona_0001.yaml"}},
            {"name": "persona-json-survey", "kwargs": {"persona_path": "persona_0002.yaml"}},
        ],
        "trial_plan": [
            {"task_index": 0, "agent_index": 0},
            {"task_index": 1, "agent_index": 1},
        ],
    }
    config_path = tmp_path / "configs" / f"{job_name}.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    service._persist_launch_meta(
        job_name,
        {
            "configPath": str(config_path.relative_to(tmp_path)),
            "jobConfig": config,
            "computeFamily": "modal",
            "executionPlane": "harbor",
            "shards": [
                {"key": "s0", "index": 0, "kind": "harbor", "personaIds": ["0001", "0002"]}
            ],
        },
    )
    service._launches[job_name] = HarborLaunchRecord(
        job_name=job_name,
        status="completed",
        config_path=str(config_path.relative_to(tmp_path)),
        started_at="2026-07-01T12:00:00Z",
    )
    for trial_name, task_index, agent_index, failed in [
        ("success-a", 0, 0, False),
        ("failed-a", 0, 0, True),
        ("success-b-1", 1, 1, False),
        ("success-b-2", 1, 1, False),
    ]:
        trial_dir = job_dir / trial_name
        trial_dir.mkdir()
        (trial_dir / "config.json").write_text(
            json.dumps({"task": config["tasks"][task_index], "agent": config["agents"][agent_index]}),
            encoding="utf-8",
        )
        (trial_dir / "result.json").write_text(
            json.dumps({"exception_info": {"exception_type": "TimeoutError"}} if failed else {}),
            encoding="utf-8",
        )

    assert service.retry_failed(job_name) == {"jobName": job_name, "retried": 1}
    meta = service._read_json(service._launch_meta_path(job_name))
    assert meta is not None
    retry_row = meta["retryShards"][0]
    assert retry_row["kind"] == "harbor"
    assert retry_row["key"].startswith("retry-")
    assert retry_row["key"] != "s0"
    retry_cfg = yaml.safe_load((tmp_path / retry_row["configPath"]).read_text())
    assert retry_cfg["n_attempts"] == 1
    assert retry_cfg["trial_plan"] == [{"task_index": 0, "agent_index": 0}]
    assert retry_cfg["tasks"][0]["path"] == "task-a"
    assert retry_cfg["agents"][0]["kwargs"]["persona_path"] == "persona_0001.yaml"
    assert retry_cfg["jobs_dir"] == (
        f"jobs/{job_name}/_generated/work/{retry_row['key']}"
    )
    assert (job_dir / "success-a" / "result.json").is_file()
    assert not (job_dir / "failed-a").exists()

    monkeypatch.setattr(
        service, "_run_with_trial_host_scoring", lambda _job_name, operation: operation()
    )
    monkeypatch.setattr(service, "_maybe_rollup_job_result", lambda _job_name: None)
    monkeypatch.setattr(service, "_maybe_run_host_verifier", lambda _job_name: None)
    monkeypatch.setattr(service, "_maybe_generate_post_run_feedback", lambda _job_name: None)
    monkeypatch.setattr(service, "_maybe_schedule_reporting", lambda *_args: None)
    fn, args, kwargs = service._executor.calls[0]
    assert fn.__name__ == "_run_harbor"
    fn(*args, **kwargs)

    assert len(commands) == 1
    assert (job_dir / "retried-a" / "result.json").is_file()
    saved_config = json.loads((job_dir / "retried-a" / "config.json").read_text())
    assert saved_config["trials_dir"] == str(Path("jobs") / job_name)
    assert (job_dir / "success-a" / "result.json").is_file()
    assert (job_dir / "success-b-1" / "result.json").is_file()
    assert (job_dir / "success-b-2" / "result.json").is_file()
    assert not (job_dir / "_generated" / "work" / retry_row["key"]).exists()
    service.shutdown()


def test_planned_harbor_fanout_merge_rewrites_trial_directory_for_resume(tmp_path):
    from harbor.models.trial.config import TrialConfig

    job_name = "planned-harbor-merge"
    jobs_dir = tmp_path / "jobs"
    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=jobs_dir,
        generated_configs_dir=tmp_path / "configs",
    )
    service._persist_launch_meta(
        job_name,
        {
            "jobConfig": {
                "jobs_dir": "jobs",
                "trial_plan": [{"task_index": 0, "agent_index": 0}],
            }
        },
    )
    staged = jobs_dir / job_name / "_generated" / "work" / "s0" / job_name
    trial_dir = staged / "trial-a"
    trial_dir.mkdir(parents=True)
    trial_config = {
        "task": {"path": "task-a"},
        "agent": {
            "name": "persona-json-survey",
            "kwargs": {"persona_path": "persona_0001.yaml"},
        },
        "trials_dir": str(staged.relative_to(tmp_path)),
    }
    (trial_dir / "config.json").write_text(json.dumps(trial_config), encoding="utf-8")
    (trial_dir / "result.json").write_text("{}", encoding="utf-8")

    service._merge_harbor_shard_tree(job_name, "s0")

    merged = TrialConfig.model_validate_json(
        (jobs_dir / job_name / "trial-a" / "config.json").read_text()
    )
    expected = TrialConfig.model_validate(
        {**trial_config, "trials_dir": str(Path("jobs") / job_name)}
    )
    assert merged == expected
    assert not staged.parent.exists()
    service.shutdown()


def test_retry_failed_no_failures_is_noop(tmp_path, monkeypatch):
    repo = tmp_path
    jobs_dir = repo / "jobs"
    job_dir = jobs_dir / "clean-job"
    trial_dir = job_dir / "trial-ok"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text("{}", encoding="utf-8")

    service = HarborJobService(
        repo_root=repo,
        jobs_dir=jobs_dir,
        generated_configs_dir=repo / "configs" / "jobs",
    )
    service._executor = _FakeExecutor()
    service._persist_launch_meta(
        "clean-job",
        {"jobConfig": {"tasks": [{"path": "task-a"}]}, "retryShards": [{"key": "s0"}]},
    )

    assert service.retry_failed("clean-job") == {"jobName": "clean-job", "retried": 0}
    assert service._executor.calls == []
    assert trial_dir.is_dir()
    service.shutdown()


def test_resolve_trial_profile_auto_survey_and_chat(tmp_path):
    repo = tmp_path
    survey_dir = repo / "application" / "tasks" / "example-survey_product-feedback"
    survey_dir.mkdir(parents=True)
    (survey_dir / "task.toml").write_text("metadata:\n  type: survey\n", encoding="utf-8")
    chat_dir = repo / "application" / "tasks" / "chat_meal-planning-nutrition"
    chat_dir.mkdir(parents=True)
    (chat_dir / "task.toml").write_text("metadata:\n  type: chat\n", encoding="utf-8")

    from backend.service.harbor_job_service import resolve_trial_profile

    assert (
        resolve_trial_profile(
            "application/tasks/example-survey_product-feedback",
            mode="auto",
            repo_root=repo,
        )
        == "json_survey"
    )
    assert (
        resolve_trial_profile(
            "application/tasks/chat_meal-planning-nutrition",
            mode="auto",
            repo_root=repo,
        )
        == "user_sim_chat"
    )
    assert (
        resolve_trial_profile(
            "application/tasks/chat_meal-planning-nutrition",
            mode="force_docker",
            repo_root=repo,
        )
        == "docker_agent"
    )


def test_resolve_agent_name_json_survey_profile(tmp_path):
    repo = tmp_path
    survey_dir = repo / "application" / "tasks" / "example-survey_product-feedback"
    survey_dir.mkdir(parents=True)
    (survey_dir / "task.toml").write_text("metadata:\n  type: survey\n", encoding="utf-8")

    from backend.service.harbor_job_service import resolve_agent_name, resolve_trial_profile

    profile = resolve_trial_profile(
        "application/tasks/example-survey_product-feedback",
        mode="auto",
        repo_root=repo,
    )
    assert profile == "json_survey"
    assert (
        resolve_agent_name(
            "application/tasks/example-survey_product-feedback",
            repo_root=repo,
            mode="auto",
            trial_profile=profile,
        )
        == "persona-json-survey"
    )
    assert (
        resolve_agent_name(
            "application/tasks/example-survey_product-feedback",
            repo_root=repo,
            mode="force_docker",
            trial_profile="docker_agent",
        )
        == "persona-claude-code"
    )


def test_resolve_agent_name_for_chat_task(tmp_path):
    repo = tmp_path
    task_dir = repo / "application" / "tasks" / "chat_meal-planning-nutrition"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        "metadata:\n  type: chat\n",
        encoding="utf-8",
    )
    from backend.service.harbor_job_service import resolve_agent_name

    assert (
        resolve_agent_name(
            "application/tasks/chat_meal-planning-nutrition",
            repo_root=repo,
            mode="auto",
        )
        == "persona-user-sim"
    )
    assert (
        resolve_agent_name(
            "application/tasks/chat_meal-planning-nutrition",
            repo_root=repo,
            mode="force_docker",
            trial_profile="docker_agent",
        )
        == "persona-claude-code"
    )


def test_launch_auto_chat_uses_local_distributed_executor(tmp_path, monkeypatch):
    repo = tmp_path
    jobs_dir = repo / "jobs"
    jobs_dir.mkdir()
    pool = repo / "persona" / "datasets" / "matraix-persona-dev-sample"
    pool.mkdir(parents=True)
    (pool / "persona_0042.yaml").write_text(
        "persona_id: '0042'\nversion: '1.0'\nsource: Nemotron\ndimensions: {}\n",
        encoding="utf-8",
    )
    task_dir = repo / "application" / "tasks" / "chat_meal-planning-nutrition"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("metadata:\n  type: chat\n", encoding="utf-8")

    calls: list[dict[str, object]] = []

    def _fake_run(command, *, cwd, env):
        calls.append({"command": list(command), "cwd": cwd, "env": dict(env)})
        return 0

    monkeypatch.setattr(
        "playground.harbor.playground._repo_root",
        lambda: repo,
    )
    service = HarborJobService(
        repo_root=repo,
        jobs_dir=jobs_dir,
        generated_configs_dir=repo / "configs" / "jobs" / "application-task-job-recipe",
        command_runner=_fake_run,
        harbor_command=("echo", "harbor"),
    )
    service._executor = _FakeExecutor()

    service.launch(
        task_path="application/tasks/chat_meal-planning-nutrition",
        persona_ids=["0042"],
        persona_model="anthropic/claude-haiku-4-5",
        execution_mode="auto",
        job_name="chat-distributed-job",
    )

    assert service._executor.calls
    assert service._executor.calls[0][0].__name__ == "_run_local_distributed"
    fn, args, kwargs = service._executor.calls[0]
    assert args[3] == "application/tasks/chat_meal-planning-nutrition"
    fn(*args, **kwargs)
    assert calls
    env = calls[0]["env"]
    assert isinstance(env, dict)
    assert env["MATRIX_CHATBOT_TASK_PATH"] == "application/tasks/chat_meal-planning-nutrition"
    pythonpath = env["PYTHONPATH"].split(":")
    assert str(repo) in pythonpath
    assert str(repo / "src") in pythonpath
    assert str(repo / "environment" / "runtime") in pythonpath
    assert str(repo / "environment" / "agents") in pythonpath
    assert str(repo / "packages" / "playground" / "src") in pythonpath
    assert str(repo / "application" / "playground") in pythonpath
    service.shutdown()


class _FakeExecutor:
    def __init__(self) -> None:
        self.calls = []

    def submit(self, fn, *args, **kwargs):
        self.calls.append((fn, args, kwargs))
        return None

    def shutdown(self, wait=False, cancel_futures=True):
        return None


def test_get_job_surfaces_reporting_queue_status(tmp_path, monkeypatch):
    repo = tmp_path
    jobs_dir = repo / "jobs"
    job_dir = jobs_dir / "demo-job"
    trial_dir = job_dir / "trial-1"
    (trial_dir / "verifier").mkdir(parents=True, exist_ok=True)
    (trial_dir / "result.json").write_text("{}", encoding="utf-8")
    (trial_dir / "config.json").write_text(
        json.dumps({"task": {"path": "application/tasks/example-task"}}),
        encoding="utf-8",
    )
    (trial_dir / "verifier" / "structured_output.json").write_text(
        json.dumps(
            {
                "presenceCheck": {"passed": True},
                "contexts": [
                    {
                        "key": "question.q1",
                        "label": "Question 1",
                        "contextType": "question_response",
                        "summaryAnalyses": [
                            {
                                "id": "question.reason_summary",
                                "title": "Reason summary",
                                "targetFacetKey": "reason",
                                "groupByFacetKey": "response",
                                "groupByMode": "categorical",
                                "summaryKind": "llm_bucket_summary",
                            }
                        ],
                        "facets": [
                            {"key": "response", "label": "Response", "role": "primary", "kind": "categorical", "value": "yes"},
                            {"key": "reason", "label": "Reason", "role": "explanation", "kind": "textual", "value": "Affordable."},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PLAYGROUND_REPORTING_ENABLE_LLM", "1")
    service = HarborJobService(
        repo_root=repo,
        jobs_dir=jobs_dir,
        generated_configs_dir=repo / "configs" / "jobs",
    )
    service._executor = _FakeExecutor()

    detail = service.get_job("demo-job")

    assert detail is not None
    # Job detail no longer embeds the full aggregation payload or schedules
    # reporting — that happens via get_job_aggregation / list_jobs.
    assert detail.get("aggregation") is None
    assert service._executor.calls == []

    aggregation = service.get_job_aggregation("demo-job")
    assert aggregation["reporting"]["status"] == "queued"
    assert len(service._executor.calls) == 1
    status_path = jobs_dir / "demo-job" / "reporting_status.json"
    assert status_path.is_file()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "queued"

    service.shutdown()


def test_get_job_aggregation_reuses_fresh_artifact(tmp_path, monkeypatch):
    repo = tmp_path
    jobs_dir = repo / "jobs"
    job_dir = jobs_dir / "cached-job"
    trial_dir = job_dir / "trial-1"
    (trial_dir / "verifier").mkdir(parents=True, exist_ok=True)
    (trial_dir / "result.json").write_text("{}", encoding="utf-8")
    (trial_dir / "config.json").write_text(
        json.dumps({"task": {"path": "application/tasks/example-task"}}),
        encoding="utf-8",
    )
    (trial_dir / "verifier" / "structured_output.json").write_text(
        json.dumps(
            {
                "presenceCheck": {"passed": True},
                "contexts": [
                    {
                        "key": "question.q1",
                        "label": "Question 1",
                        "contextType": "question_response",
                        "facets": [
                            {
                                "key": "response",
                                "label": "Response",
                                "role": "primary",
                                "kind": "categorical",
                                "value": "yes",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("PLAYGROUND_REPORTING_ENABLE_LLM", raising=False)
    service = HarborJobService(
        repo_root=repo,
        jobs_dir=jobs_dir,
        generated_configs_dir=repo / "configs" / "jobs",
    )
    service._executor = _FakeExecutor()

    first = service.get_job_aggregation("cached-job")
    generated_at = first["generatedAt"]
    artifact_mtime = (job_dir / "aggregation.json").stat().st_mtime_ns

    second = service.get_job_aggregation("cached-job")
    assert second["generatedAt"] == generated_at
    assert (job_dir / "aggregation.json").stat().st_mtime_ns == artifact_mtime

    # New completed trial makes the cache stale and forces a rebuild.
    trial_2 = job_dir / "trial-2"
    (trial_2 / "verifier").mkdir(parents=True, exist_ok=True)
    (trial_2 / "result.json").write_text("{}", encoding="utf-8")
    (trial_2 / "verifier" / "structured_output.json").write_text(
        (trial_dir / "verifier" / "structured_output.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    third = service.get_job_aggregation("cached-job")
    assert third["coverage"]["trialCount"] == 2
    assert third["coverage"]["completedTrials"] == 2
    assert (job_dir / "aggregation.json").stat().st_mtime_ns > artifact_mtime

    service.shutdown()


def test_launch_ios_cua_uses_use_computer_environment(tmp_path, monkeypatch):
    repo = tmp_path
    jobs_dir = repo / "jobs"
    jobs_dir.mkdir()
    pool = repo / "persona" / "datasets" / "matraix-persona-dev-sample"
    pool.mkdir(parents=True)
    (pool / "persona_0020.yaml").write_text(
        "persona_id: '0020'\nversion: '1.0'\nsource: OASIS\ndimensions: {}\n",
        encoding="utf-8",
    )
    task_dir = repo / "application" / "tasks" / "example-computer-use-ios_photo-access-review"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("metadata:\n  type: mobile\n", encoding="utf-8")

    def _fake_run(command, *, cwd, env):
        return 0

    monkeypatch.setattr(
        "playground.harbor.playground._repo_root",
        lambda: repo,
    )
    service = HarborJobService(
        repo_root=repo,
        jobs_dir=jobs_dir,
        generated_configs_dir=repo / "configs" / "jobs" / "application-task-job-recipe",
        command_runner=_fake_run,
        harbor_command=("echo", "harbor"),
    )

    job_name = service.launch(
        task_path="application/tasks/example-computer-use-ios_photo-access-review",
        persona_ids=["0020"],
        persona_model="anthropic/claude-sonnet-4-6",
        cua_backend="ios",
        job_name="ios-cua-job",
    )
    assert job_name == "ios-cua-job"
    text = (
        repo / "configs" / "jobs" / "application-task-job-recipe" / "ios-cua-job.yaml"
    ).read_text(encoding="utf-8")
    assert "type: use-computer" in text
    assert "platform: ios" in text
    assert "cua_backend: ios" in text
    assert "max_steps: 100" in text
    assert "max_turns:" not in text
    service.shutdown()


def test_launch_docker_cua_injects_max_turns_not_max_steps(tmp_path, monkeypatch):
    repo = tmp_path
    jobs_dir = repo / "jobs"
    jobs_dir.mkdir()
    pool = repo / "persona" / "datasets" / "matraix-persona-dev-sample"
    pool.mkdir(parents=True)
    (pool / "persona_0020.yaml").write_text(
        "persona_id: '0020'\nversion: '1.0'\nsource: OASIS\ndimensions: {}\n",
        encoding="utf-8",
    )
    task_dir = repo / "application" / "tasks" / "example-computer-use-linux_note-to-csv"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("metadata:\n  type: computer\n", encoding="utf-8")

    def _fake_run(command, *, cwd, env):
        return 0

    monkeypatch.setattr(
        "playground.harbor.playground._repo_root",
        lambda: repo,
    )
    service = HarborJobService(
        repo_root=repo,
        jobs_dir=jobs_dir,
        generated_configs_dir=repo / "configs" / "jobs" / "application-task-job-recipe",
        command_runner=_fake_run,
        harbor_command=("echo", "harbor"),
    )

    job_name = service.launch(
        task_path="application/tasks/example-computer-use-linux_note-to-csv",
        persona_ids=["0020"],
        persona_model="anthropic/claude-sonnet-4-6",
        cua_backend="docker",
        job_name="docker-cua-job",
    )
    assert job_name == "docker-cua-job"
    text = (
        repo / "configs" / "jobs" / "application-task-job-recipe" / "docker-cua-job.yaml"
    ).read_text(encoding="utf-8")
    assert "type: docker" in text
    assert "cua_backend: docker" in text
    assert "max_turns: 100" in text
    assert "max_steps:" not in text
    service.shutdown()


def test_run_harbor_passes_yes_to_skip_host_env_prompt(tmp_path):
    repo = tmp_path
    jobs_dir = repo / "jobs"
    jobs_dir.mkdir()
    config_path = repo / "configs" / "jobs" / "demo-job.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("job_name: demo-job\n", encoding="utf-8")

    calls: list[list[str]] = []

    def _fake_run(command, *, cwd, env):
        calls.append(list(command))
        return 0

    service = HarborJobService(
        repo_root=repo,
        jobs_dir=jobs_dir,
        generated_configs_dir=repo / "configs" / "jobs",
        command_runner=_fake_run,
        harbor_command=("echo", "harbor", "run"),
    )
    with service._guard:
        service._launches["demo-job"] = HarborLaunchRecord(
            job_name="demo-job",
            config_path="configs/jobs/demo-job.yaml",
        )

    service._run_harbor("demo-job", config_path)
    assert calls == [["echo", "harbor", "run", "--yes", "-c", "configs/jobs/demo-job.yaml"]]
    service.shutdown()


def test_run_harbor_scores_trial_when_result_appears(tmp_path, monkeypatch):
    import time

    repo = tmp_path
    jobs_dir = repo / "jobs"
    jobs_dir.mkdir()
    config_path = repo / "configs" / "jobs" / "demo-job.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("job_name: demo-job\n", encoding="utf-8")

    scored: list[str] = []

    def _fake_run(command, *, cwd, env):
        del command, cwd, env
        trial = jobs_dir / "demo-job" / "trial-x"
        trial.mkdir(parents=True)
        (trial / "config.json").write_text("{}", encoding="utf-8")
        (trial / "result.json").write_text("{}", encoding="utf-8")
        # Let the watcher poll once before Harbor "exits".
        time.sleep(0.25)
        return 0

    service = HarborJobService(
        repo_root=repo,
        jobs_dir=jobs_dir,
        generated_configs_dir=repo / "configs" / "jobs",
        command_runner=_fake_run,
        harbor_command=("echo", "harbor", "run"),
        _host_score_poll_sec=0.05,
        _host_score_join_timeout_sec=5.0,
    )
    monkeypatch.setattr(
        service,
        "_score_finished_trial_on_host",
        lambda trial_dir: scored.append(trial_dir.name),
    )
    monkeypatch.setattr(service, "_maybe_run_host_verifier", lambda job_name: None)
    monkeypatch.setattr(service, "_maybe_generate_post_run_feedback", lambda job_name: None)
    monkeypatch.setattr(
        service, "_maybe_schedule_reporting", lambda job_name, job_dir: None
    )
    with service._guard:
        service._launches["demo-job"] = HarborLaunchRecord(
            job_name="demo-job",
            config_path="configs/jobs/demo-job.yaml",
        )

    service._run_harbor("demo-job", config_path)

    assert scored == ["trial-x"]
    service.shutdown()


def test_score_finished_trial_on_host_runs_verifier_then_feedback(tmp_path, monkeypatch):
    repo = tmp_path
    trial_dir = repo / "jobs" / "demo-job" / "trial-a"
    trial_dir.mkdir(parents=True)
    (trial_dir / "config.json").write_text("{}", encoding="utf-8")
    (trial_dir / "result.json").write_text("{}", encoding="utf-8")

    order: list[str] = []
    monkeypatch.setattr(
        "playground.host_verifier.maybe_run_host_verifier",
        lambda *, repo_root, trial_dir, timeout_sec=None: order.append("verify"),
    )
    monkeypatch.setattr(
        "playground.post_run_feedback.maybe_write_trial_user_feedback",
        lambda *, repo_root, trial_dir: order.append("feedback"),
    )
    service = HarborJobService(
        repo_root=repo,
        jobs_dir=repo / "jobs",
        generated_configs_dir=repo / "configs" / "jobs",
    )

    service._score_finished_trial_on_host(trial_dir)

    assert order == ["verify", "feedback"]
    service.shutdown()


def test_trial_live_stage_from_artifacts(tmp_path):
    from backend.service.harbor_job_service import (
        _resolve_trial_stage,
        _stage_from_phase,
        _trial_live_stage,
    )

    trial_dir = tmp_path / "trial-a"
    assert _trial_live_stage(trial_dir) == "queued"

    trial_dir.mkdir()
    (trial_dir / "config.json").write_text("{}", encoding="utf-8")
    assert _trial_live_stage(trial_dir) == "starting_env"

    agent_dir = trial_dir / "agent"
    agent_dir.mkdir()
    (agent_dir / "setup").mkdir()
    assert _trial_live_stage(trial_dir) == "starting_env"

    (agent_dir / "trajectory.json").write_text("[]", encoding="utf-8")
    assert _trial_live_stage(trial_dir) == "agent_running"

    verifier_dir = trial_dir / "verifier"
    verifier_dir.mkdir()
    (verifier_dir / "reward.txt").write_text("1.0", encoding="utf-8")
    assert _trial_live_stage(trial_dir) == "verifying"

    (trial_dir / "result.json").write_text("{}", encoding="utf-8")
    assert _trial_live_stage(trial_dir) is None

    assert _stage_from_phase("persona_thinking") == "agent_running"
    assert _stage_from_phase("harbor_collecting_artifacts") == "verifying"
    assert _resolve_trial_stage(trial_dir, phase="persona_kickoff", completed=False) == "agent_running"


def test_get_job_status_incremental(tmp_path):
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "pg-batch"
    job_dir.mkdir(parents=True)
    for name in ("t0", "t1", "t2"):
        (job_dir / name).mkdir()

    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=jobs_dir,
        generated_configs_dir=tmp_path / "generated",
    )

    first = service.get_job_status("pg-batch", since=0)
    assert first["full"] is True
    assert first["trialCount"] == 3
    assert first["statuses"] == [0, 0, 0]
    assert first["counts"] == {"pending": 3, "running": 0, "done": 0, "error": 0}

    # t0 finishes successfully -> incremental delta only.
    (job_dir / "t0" / "result.json").write_text("{}", encoding="utf-8")
    delta = service.get_job_status("pg-batch", since=first["version"])
    assert delta["full"] is False
    assert delta["changes"] == [[0, 2]]
    assert delta["counts"]["done"] == 1

    # t1 fails.
    (job_dir / "t1" / "result.json").write_text(
        json.dumps({"exception_info": {"exception_message": "boom"}}),
        encoding="utf-8",
    )
    delta2 = service.get_job_status("pg-batch", since=delta["version"])
    assert delta2["full"] is False
    assert delta2["changes"] == [[1, 3]]
    assert delta2["counts"] == {"pending": 1, "running": 0, "done": 1, "error": 1}

    # A stale/zero cursor always yields a fresh full snapshot.
    full = service.get_job_status("pg-batch", since=0)
    assert full["full"] is True
    assert full["statuses"] == [2, 3, 0]


def test_get_job_status_uses_live_overlay_without_trial_trees(tmp_path):
    from backend.service.harbor_job_service import STATUS_CODE_DONE, STATUS_CODE_RUNNING
    from backend.service.modal_host_job import apply_live_overlay

    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "pg-batch"
    job_dir.mkdir(parents=True)
    apply_live_overlay(job_dir, {"survey__a": "done", "survey__b": "running"})

    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=jobs_dir,
        generated_configs_dir=tmp_path / "generated",
    )
    snapshot = service.get_job_status("pg-batch", since=0)
    assert snapshot["trialNames"] == ["survey__a", "survey__b"]
    assert snapshot["statuses"][0] == STATUS_CODE_DONE
    assert snapshot["statuses"][1] == STATUS_CODE_RUNNING
    assert snapshot["counts"]["done"] == 1
    assert snapshot["counts"]["running"] == 1


def test_get_job_live_uses_live_overlay_without_trial_trees(tmp_path):
    from backend.service.modal_host_job import apply_live_overlay

    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "pg-batch"
    job_dir.mkdir(parents=True)
    apply_live_overlay(job_dir, {"survey__a": "done", "survey__b": "running"})

    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=jobs_dir,
        generated_configs_dir=tmp_path / "generated",
    )
    live = service.get_job_live("pg-batch")
    by_name = {row["trialName"]: row for row in live["trials"]}
    assert by_name["survey__a"]["completed"] is True
    assert by_name["survey__a"]["succeeded"] is True
    assert by_name["survey__b"]["completed"] is False
    assert by_name["survey__b"]["stage"] == "agent_running"
    assert live["completedTrials"] == 1


def test_get_trial_events_empty_before_trial_tree(tmp_path):
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "pg-batch"
    job_dir.mkdir(parents=True)

    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=jobs_dir,
        generated_configs_dir=tmp_path / "generated",
    )
    payload = service.get_trial_events("pg-batch", "survey__a")
    assert payload == {"events": [], "offset": 0}


def test_get_trial_web_trace_empty_before_trial_tree(tmp_path):
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "pg-web"
    job_dir.mkdir(parents=True)

    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=jobs_dir,
        generated_configs_dir=tmp_path / "generated",
    )
    payload = service.get_trial_web_trace("pg-web", "web__a")
    assert payload == {"trace": {"events": [], "raw": {}}}


def test_trial_live_screenshot_falls_back_to_disk(tmp_path):
    jobs_dir = tmp_path / "jobs"
    trial = jobs_dir / "pg-web" / "web__a"
    images = trial / "agent" / "images"
    images.mkdir(parents=True)
    (images / "step_001.png").write_bytes(b"png-bytes")

    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=jobs_dir,
        generated_configs_dir=tmp_path / "generated",
    )
    assert service.trial_live_screenshot("pg-web", "web__a") == b"png-bytes"


def test_get_job_status_running_marker(tmp_path):
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "pg-batch"
    job_dir.mkdir(parents=True)
    (job_dir / "t0").mkdir()
    (job_dir / "t0" / "config.json").write_text("{}", encoding="utf-8")

    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=jobs_dir,
        generated_configs_dir=tmp_path / "generated",
    )
    snapshot = service.get_job_status("pg-batch", since=0)
    assert snapshot["statuses"] == [1]
    assert snapshot["counts"]["running"] == 1


def test_list_jobs_skips_aggregation_rebuild_for_completed_reporting(tmp_path, monkeypatch):
    import json

    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "done-job"
    job_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "finished_at": "2026-07-01T12:00:00Z",
                "n_total_trials": 1008,
                "stats": {"n_completed_trials": 1008, "n_errored_trials": 0},
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "aggregation.json").write_text(
        json.dumps(
            {
                "schemaVersion": "1.0",
                "artifactType": "job_aggregation",
                "coverage": {
                    "trialCount": 1008,
                    "completedTrials": 1008,
                    "pendingTrials": 0,
                },
                "fields": [],
                "contexts": [],
                "reporting": {"status": "completed", "totalUnits": 0, "completedUnits": 0},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PLAYGROUND_REPORTING_ENABLE_LLM", "1")
    calls: list[str] = []

    def _boom(*_args, **_kwargs):
        calls.append("build")
        raise AssertionError("list_jobs must not rebuild aggregation for terminal reporting")

    monkeypatch.setattr(
        "backend.service.harbor_job_service.build_job_aggregation",
        _boom,
    )
    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=jobs_dir,
        generated_configs_dir=tmp_path / "configs",
    )
    rows = service.list_jobs()
    assert len(rows) == 1
    assert rows[0]["jobName"] == "done-job"
    assert rows[0]["trialCount"] == 1008
    assert rows[0]["completedTrials"] == 1008
    assert rows[0]["status"] == "success"
    assert calls == []
    service.shutdown()


def test_list_jobs_reports_success_and_failed_status(tmp_path):
    import json

    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    success_job = jobs_dir / "job-success"
    success_job.mkdir()
    (success_job / "trial-a").mkdir()
    (success_job / "trial-a" / "result.json").write_text("{}", encoding="utf-8")
    (success_job / "result.json").write_text(
        json.dumps(
            {
                "finished_at": "2026-07-01T12:00:00Z",
                "n_total_trials": 1,
                "stats": {"n_completed_trials": 1, "n_errored_trials": 0},
            }
        ),
        encoding="utf-8",
    )

    failed_job = jobs_dir / "job-failed"
    failed_job.mkdir()
    (failed_job / "trial-a").mkdir()
    (failed_job / "trial-a" / "result.json").write_text("{}", encoding="utf-8")
    (failed_job / "result.json").write_text(
        json.dumps(
            {
                "finished_at": "2026-07-01T12:05:00Z",
                "stats": {"n_errored_trials": 1},
            }
        ),
        encoding="utf-8",
    )

    running_job = jobs_dir / "job-running"
    running_job.mkdir()
    (running_job / "trial-a").mkdir()

    # An orphaned job (trial dirs present but incomplete, and no in-memory launch
    # record because the backend was restarted) is terminal, not "running".
    orphaned_job = jobs_dir / "job-orphaned"
    orphaned_job.mkdir()
    (orphaned_job / "trial-a").mkdir()

    # A fully completed cohort without a job-level result.json still resolves to
    # a terminal status from disk instead of a stale "running".
    completed_job = jobs_dir / "job-completed"
    completed_job.mkdir()
    (completed_job / "trial-a").mkdir()
    (completed_job / "trial-a" / "result.json").write_text("{}", encoding="utf-8")

    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=jobs_dir,
        generated_configs_dir=tmp_path / "configs",
    )
    # Only "job-running" is actively driven by this process.
    service._launches["job-running"] = HarborLaunchRecord(
        job_name="job-running",
        status="running",
    )
    rows = {row["jobName"]: row for row in service.list_jobs()}

    assert rows["job-success"]["status"] == "success"
    assert rows["job-success"]["failedTrials"] == 0
    assert rows["job-failed"]["status"] == "failed"
    assert rows["job-failed"]["failedTrials"] == 1
    assert rows["job-running"]["status"] == "running"
    assert rows["job-orphaned"]["status"] == "failed"
    assert rows["job-completed"]["status"] == "success"
    service.shutdown()


def _stale_iso(now_ts: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(
        now_ts - EXTERNAL_RUN_STALE_AFTER_SEC - 60, tz=timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fresh_iso(now_ts: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(now_ts - 30, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_job_list_status_external_live_stats_are_running():
    now = 1_700_000_000.0
    status, failed = _job_list_status(
        trial_count=2,
        completed_trials=0,
        failed_trials_on_disk=0,
        job_result={
            "updated_at": _fresh_iso(now),
            "stats": {"n_running_trials": 1, "n_pending_trials": 1, "n_errored_trials": 0},
        },
        launch=None,
        activity_ts=now - 30,
        now_ts=now,
    )
    assert status == "running"
    assert failed == 0


def test_job_list_status_external_pending_only_is_running():
    now = 1_700_000_000.0
    status, _failed = _job_list_status(
        trial_count=4,
        completed_trials=0,
        failed_trials_on_disk=0,
        job_result={
            "updated_at": _fresh_iso(now),
            "stats": {"n_running_trials": 0, "n_pending_trials": 4},
        },
        launch=None,
        activity_ts=now - 5,
        now_ts=now,
    )
    assert status == "running"


def test_job_list_status_stale_in_flight_stats_are_failed_not_running():
    now = 1_700_000_000.0
    status, failed = _job_list_status(
        trial_count=2,
        completed_trials=0,
        failed_trials_on_disk=0,
        job_result={
            "updated_at": _stale_iso(now),
            "stats": {"n_running_trials": 1, "n_pending_trials": 1, "n_errored_trials": 0},
        },
        launch=None,
        activity_ts=now - EXTERNAL_RUN_STALE_AFTER_SEC - 120,
        now_ts=now,
    )
    assert status == "failed"
    assert failed == 0


def test_job_list_status_in_memory_launch_stays_running_even_if_disk_is_stale():
    now = 1_700_000_000.0
    status, failed = _job_list_status(
        trial_count=2,
        completed_trials=0,
        failed_trials_on_disk=0,
        job_result={
            "updated_at": _stale_iso(now),
            "stats": {"n_running_trials": 1, "n_pending_trials": 1},
        },
        launch=HarborLaunchRecord(job_name="still-here", status="running"),
        activity_ts=now - EXTERNAL_RUN_STALE_AFTER_SEC - 120,
        now_ts=now,
    )
    assert status == "running"
    assert failed == 0


def test_list_jobs_external_run_uses_freshness(tmp_path):
    import os
    from datetime import datetime, timezone

    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    now = datetime.now(timezone.utc).timestamp()
    stale = now - EXTERNAL_RUN_STALE_AFTER_SEC - 120

    live = jobs_dir / "job-external-live"
    live.mkdir()
    (live / "trial-a").mkdir()
    (live / "result.json").write_text(
        json.dumps(
            {
                "updated_at": datetime.fromtimestamp(now - 10, tz=timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "stats": {"n_running_trials": 1, "n_pending_trials": 0},
            }
        ),
        encoding="utf-8",
    )

    dead = jobs_dir / "job-external-dead"
    dead.mkdir()
    (dead / "trial-a").mkdir()
    (dead / "result.json").write_text(
        json.dumps(
            {
                "updated_at": datetime.fromtimestamp(stale, tz=timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "stats": {"n_running_trials": 1, "n_pending_trials": 1},
            }
        ),
        encoding="utf-8",
    )
    for path in (dead, dead / "result.json", dead / "trial-a"):
        os.utime(path, (stale, stale))

    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=jobs_dir,
        generated_configs_dir=tmp_path / "configs",
    )
    rows = {row["jobName"]: row for row in service.list_jobs()}
    assert rows["job-external-live"]["status"] == "running"
    assert rows["job-external-dead"]["status"] == "failed"
    service.shutdown()


def test_list_jobs_external_run_uses_live_overlay_freshness(tmp_path):
    import os
    from datetime import datetime, timezone

    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    now = datetime.now(timezone.utc).timestamp()
    stale = now - EXTERNAL_RUN_STALE_AFTER_SEC - 120

    job_dir = jobs_dir / "job-modal-overlay"
    generated = job_dir / "_generated"
    generated.mkdir(parents=True)
    (job_dir / "trial-a").mkdir()
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "updated_at": datetime.fromtimestamp(stale, tz=timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "stats": {"n_running_trials": 1, "n_pending_trials": 0},
            }
        ),
        encoding="utf-8",
    )
    (generated / "live_overlay.json").write_text("{}", encoding="utf-8")
    for path in (job_dir, job_dir / "result.json", job_dir / "trial-a"):
        os.utime(path, (stale, stale))

    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=jobs_dir,
        generated_configs_dir=tmp_path / "configs",
    )
    rows = {row["jobName"]: row for row in service.list_jobs()}
    assert rows["job-modal-overlay"]["status"] == "running"
    service.shutdown()


def test_list_jobs_counts_overlay_and_expected_agents(tmp_path):
    from backend.service.modal_host_job import apply_live_overlay

    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "pg-live"
    job_dir.mkdir(parents=True)
    (job_dir / "config.json").write_text(
        json.dumps(
            {
                "agents": [
                    {"kwargs": {"persona_path": "pool/persona_a.yaml"}},
                    {"kwargs": {"persona_path": "pool/persona_b.yaml"}},
                    {"kwargs": {"persona_path": "pool/persona_c.yaml"}},
                    {"kwargs": {"persona_path": "pool/persona_d.yaml"}},
                ]
            }
        ),
        encoding="utf-8",
    )
    apply_live_overlay(job_dir, {"survey__a": "done", "survey__b": "running"})

    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=jobs_dir,
        generated_configs_dir=tmp_path / "configs",
    )
    service._launches["pg-live"] = HarborLaunchRecord(
        job_name="pg-live",
        status="running",
    )
    rows = {row["jobName"]: row for row in service.list_jobs()}
    assert rows["pg-live"]["trialCount"] == 4
    assert rows["pg-live"]["completedTrials"] == 1
    assert rows["pg-live"]["status"] == "running"
    service.shutdown()


def test_expected_trial_count_uses_plan_and_attempts_without_changing_legacy(tmp_path):
    job_dir = tmp_path / "jobs" / "planned"
    job_dir.mkdir(parents=True)
    config = {
        "agents": [{}, {}, {}],
        "tasks": [{}, {}],
        "n_attempts": 2,
        "trial_plan": [
            {"task_index": 0, "agent_index": 0},
            {"task_index": 1, "agent_index": 2},
        ],
    }
    config_path = job_dir / "config.json"
    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=tmp_path / "jobs",
        generated_configs_dir=tmp_path / "configs",
    )

    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert service._expected_trial_count("planned", job_dir) == 4
    config_path.write_text(json.dumps({**config, "trial_plan": []}), encoding="utf-8")
    assert service._expected_trial_count("planned", job_dir) == 3
    config_path.write_text(
        json.dumps({key: value for key, value in config.items() if key != "trial_plan"}),
        encoding="utf-8",
    )
    assert service._expected_trial_count("planned", job_dir) == 3
    service.shutdown()


def test_list_jobs_discounts_missing_reward_when_artifacts_exist(tmp_path):
    import json

    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    exp_job = jobs_dir / "exp-survey-onboarding-framing-1e9fd26e-c0"
    trial = exp_job / "survey-onboarding-framing__persona-a"
    (trial / "verifier").mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "exception_info": {
                    "exception_type": "RewardFileNotFoundError",
                    "exception_message": "Reward file not found",
                }
            }
        ),
        encoding="utf-8",
    )
    (trial / "verifier" / "structured_output.json").write_text("{}", encoding="utf-8")
    (exp_job / "result.json").write_text(
        json.dumps(
            {
                "finished_at": "2026-08-25T12:00:00Z",
                "n_total_trials": 1,
                "stats": {"n_completed_trials": 1, "n_errored_trials": 1},
            }
        ),
        encoding="utf-8",
    )

    pg_job = jobs_dir / "pg-survey-onboarding-framing-aa11bb22"
    pg_trial = pg_job / "trial-a"
    pg_trial.mkdir(parents=True)
    (pg_trial / "result.json").write_text(
        json.dumps(
            {
                "exception_info": {
                    "exception_type": "RuntimeError",
                    "exception_message": "auth failed",
                }
            }
        ),
        encoding="utf-8",
    )
    (pg_job / "result.json").write_text(
        json.dumps(
            {
                "finished_at": "2026-08-25T12:05:00Z",
                "n_total_trials": 1,
                "stats": {"n_completed_trials": 1, "n_errored_trials": 1},
            }
        ),
        encoding="utf-8",
    )

    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=jobs_dir,
        generated_configs_dir=tmp_path / "configs",
    )
    rows = {row["jobName"]: row for row in service.list_jobs()}
    assert rows[exp_job.name]["source"] == "experiment"
    assert rows[exp_job.name]["status"] == "success"
    assert rows[exp_job.name]["failedTrials"] == 0
    assert rows[pg_job.name]["source"] == "playground"
    assert rows[pg_job.name]["status"] == "failed"
    detail = service.get_job(exp_job.name)
    assert detail is not None
    assert detail["trials"][0]["succeeded"] is True
    assert detail["trials"][0]["error"] is None
    service.shutdown()


def test_list_jobs_reports_application_type_from_generated_config(tmp_path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()

    survey_job = jobs_dir / "pg-survey-job"
    survey_job.mkdir()
    (survey_job / "trial-a").mkdir()

    task_dir = tmp_path / "application" / "tasks" / "example-survey"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("metadata:\n  type: survey\n", encoding="utf-8")

    (configs_dir / "pg-survey-job.yaml").write_text(
        "# Generated by Playground POST /api/harbor/jobs\n"
        "# Task: application/tasks/example-survey\n"
        "job_name: pg-survey-job\n",
        encoding="utf-8",
    )

    service = HarborJobService(
        repo_root=tmp_path,
        jobs_dir=jobs_dir,
        generated_configs_dir=configs_dir,
    )
    rows = {row["jobName"]: row for row in service.list_jobs()}
    assert rows["pg-survey-job"]["applicationType"] == "survey"
    service.shutdown()


def test_launch_web_cli_stages_shared_web_cli_environment(tmp_path, monkeypatch):
    repo = tmp_path
    jobs_dir = repo / "jobs"
    jobs_dir.mkdir()
    task_dir = repo / "application" / "tasks" / "example-web-playwright_quote-choice"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        "\n".join(
            [
                "[metadata]",
                'type = "web"',
                "",
                "[environment]",
                'definition = "application/shared-web-playwright"',
            ]
        ),
        encoding="utf-8",
    )
    (repo / "persona" / "datasets" / "matraix-persona-dev-sample").mkdir(parents=True)
    (repo / "persona" / "datasets" / "matraix-persona-dev-sample" / "persona_0001.yaml").write_text(
        "persona_id: '0001'\nversion: '1.0'\nsource: Nemotron\ndimensions: {}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("playground.harbor.playground._repo_root", lambda: repo)
    service = HarborJobService(
        repo_root=repo,
        jobs_dir=jobs_dir,
        generated_configs_dir=repo / "configs" / "jobs" / "application-task-job-recipe",
        command_runner=lambda *args, **kwargs: 0,
        harbor_command=("echo", "harbor"),
    )
    service._executor = _FakeExecutor()

    job_name = service.launch(
        task_path="application/tasks/example-web-playwright_quote-choice",
        persona_ids=["0001"],
        persona_pool="persona/datasets/matraix-persona-dev-sample",
        agent_name="persona-codex",
        persona_model="openai/gpt-4o-mini",
        job_name="web-cli-job",
        execution_mode="auto",
    )
    assert job_name == "web-cli-job"
    config_path = repo / "configs" / "jobs" / "application-task-job-recipe" / "web-cli-job.yaml"
    text = config_path.read_text(encoding="utf-8")
    assert "web_cli_staged_tasks/example-web-playwright_quote-choice" in text
    staged_root = repo / "data" / "cache" / "playground" / "web_cli_staged_tasks"
    staged_tomls = list(staged_root.rglob("task.toml"))
    assert staged_tomls
    assert 'application/shared-web-cli' in staged_tomls[0].read_text(encoding="utf-8")
    service.shutdown()
