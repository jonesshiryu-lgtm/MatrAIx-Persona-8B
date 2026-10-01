"""Tests for application Harbor job config helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from harbor.models.job.config import JobConfig
from matraix.application_job import (
    build_application_job_config,
    build_condition_assignment_manifest,
    resolve_job_environment,
    resolve_harbor_task_path,
    resolve_persona_entries,
)


def test_resolve_persona_entries_from_pool(tmp_path: Path) -> None:
    repo = tmp_path
    pool = repo / "persona" / "datasets" / "matraix-persona-dev-sample"
    pool.mkdir(parents=True)
    (pool / "persona_0042.yaml").write_text(
        "persona_id: '0042'\nversion: '1.0'\nsource: Nemotron\ndimensions: {}\n",
        encoding="utf-8",
    )

    chosen = resolve_persona_entries(
        ["42", "0042"],
        persona_pool="persona/datasets/matraix-persona-dev-sample",
        repo_root=repo,
    )
    assert len(chosen) == 1
    assert chosen[0]["persona_id"] == "0042"


def test_resolve_persona_entries_accepts_persona_prefixed_ids(tmp_path: Path) -> None:
    repo = tmp_path
    pool = repo / "persona" / "datasets" / "matraix-persona-dev-sample"
    pool.mkdir(parents=True)
    (pool / "persona_0042.yaml").write_text(
        "persona_id: '0042'\nversion: '1.0'\nsource: Nemotron\ndimensions: {}\n",
        encoding="utf-8",
    )

    chosen = resolve_persona_entries(
        ["persona_0042"],
        persona_pool="persona/datasets/matraix-persona-dev-sample",
        repo_root=repo,
    )
    assert len(chosen) == 1
    assert chosen[0]["path"] == "persona/datasets/matraix-persona-dev-sample/persona_0042.yaml"


def test_resolve_persona_entries_unknown_includes_pool(tmp_path: Path) -> None:
    repo = tmp_path
    pool = repo / "persona" / "datasets" / "matraix-persona-dev-sample"
    pool.mkdir(parents=True)
    (pool / "persona_0042.yaml").write_text(
        "persona_id: '0042'\nversion: '1.0'\nsource: Nemotron\ndimensions: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"unknown persona wiki-e690edb701e2 in pool persona/datasets/matraix-persona-dev-sample",
    ):
        resolve_persona_entries(
            ["wiki-e690edb701e2"],
            persona_pool="persona/datasets/matraix-persona-dev-sample",
            repo_root=repo,
        )


def test_build_application_job_config_with_explicit_persona_ids(tmp_path: Path) -> None:
    repo = tmp_path
    pool = repo / "persona" / "datasets" / "matraix-persona-dev-sample"
    pool.mkdir(parents=True)
    (pool / "persona_0001.yaml").write_text(
        "persona_id: '0001'\nversion: '1.0'\nsource: Nemotron\ndimensions: {}\n",
        encoding="utf-8",
    )

    job = build_application_job_config(
        {
            "name": "explicit-job",
            "task": "application/tasks/example-survey_product-feedback",
            "persona_pool": "persona/datasets/matraix-persona-dev-sample",
            "persona_ids": ["0001"],
            "execution_mode": "auto",
            "trial_profile": "json_survey",
            "agent": {"name": "persona-claude-code", "model_name": "anthropic/claude-haiku-4-5"},
            "job": {"job_name": "explicit-job", "jobs_dir": "jobs"},
        },
        repo_root=repo,
    )
    meta = job.pop("_job_meta")
    assert meta["selected_persona_ids"] == ["0001"]
    assert meta["execution_mode"] == "auto"
    assert meta["trial_profile"] == "json_survey"
    assert len(job["agents"]) == 1
    assert job["agents"][0]["kwargs"]["persona_path"].endswith("persona_0001.yaml")


def test_resolve_job_environment_auto_native_profiles() -> None:
    assert resolve_job_environment(execution_mode="auto", trial_profile="json_survey") == {
        "type": "host",
        "delete": True,
    }
    assert resolve_job_environment(execution_mode="auto", trial_profile="user_sim_chat") == {
        "type": "host",
        "delete": True,
    }
    assert resolve_job_environment(execution_mode="auto", trial_profile="docker_agent") == {
        "type": "docker",
        "delete": True,
    }
    assert resolve_job_environment(execution_mode="force_docker", trial_profile="json_survey") == {
        "type": "docker",
        "delete": True,
    }


def test_resolve_job_environment_use_computer_for_macos_and_ios() -> None:
    assert resolve_job_environment(
        execution_mode="auto",
        trial_profile="docker_agent",
        cua_backend="macos",
    ) == {"type": "use-computer", "delete": True}
    assert resolve_job_environment(
        execution_mode="auto",
        trial_profile="docker_agent",
        cua_backend="ios",
    ) == {"type": "use-computer", "delete": True, "kwargs": {"platform": "ios"}}
    assert resolve_job_environment(
        execution_mode="auto",
        trial_profile="docker_agent",
        cua_backend="docker",
    ) == {"type": "docker", "delete": True}


def test_resolve_job_environment_compute_family(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "MATRIX_GKE_CLUSTER",
        "MATRIX_GKE_REGION",
        "MATRIX_GKE_REGISTRY",
        "GCP_PROJECT",
        "GOOGLE_CLOUD_PROJECT",
        "MATRIX_GCP_PROJECT",
    ):
        monkeypatch.delenv(name, raising=False)
    assert resolve_job_environment(
        execution_mode="auto",
        trial_profile="json_survey",
        compute_family="local",
    ) == {"type": "host", "delete": True}
    assert resolve_job_environment(
        execution_mode="auto",
        trial_profile="json_survey",
        compute_family="modal",
    ) == {"type": "host", "delete": True}
    assert resolve_job_environment(
        execution_mode="auto",
        trial_profile="user_sim_chat",
        compute_family="modal",
    ) == {"type": "host", "delete": True}
    assert resolve_job_environment(
        execution_mode="auto",
        trial_profile="docker_agent",
        compute_family="modal",
    ) == {"type": "docker", "delete": True}
    assert resolve_job_environment(
        execution_mode="auto",
        trial_profile="docker_agent",
        compute_family="gcp",
    ) == {"type": "gke", "delete": True}


def test_build_application_job_config_macos_cua_uses_use_computer(tmp_path: Path) -> None:
    repo = tmp_path
    pool = repo / "persona" / "datasets" / "matraix-persona-dev-sample"
    pool.mkdir(parents=True)
    (pool / "persona_0001.yaml").write_text(
        "persona_id: '0001'\nversion: '1.0'\nsource: Nemotron\ndimensions: {}\n",
        encoding="utf-8",
    )

    job = build_application_job_config(
        {
            "name": "macos-cua",
            "task": "application/tasks/example-computer-use-macos_calendar-reminder-handoff",
            "persona_pool": "persona/datasets/matraix-persona-dev-sample",
            "persona_ids": ["0001"],
            "execution_mode": "auto",
            "trial_profile": "docker_agent",
            "cua_backend": "macos",
            "agent": {"name": "persona-computer-1", "model_name": "anthropic/claude-haiku-4-5"},
            "job": {"job_name": "macos-cua", "jobs_dir": "jobs"},
        },
        repo_root=repo,
    )
    job.pop("_job_meta")
    assert job["environment"] == {"type": "use-computer", "delete": True}
    assert "cua_backend" not in job["agents"][0]["kwargs"]  # injected later by harbor_job_service


def test_build_application_job_config_auto_survey_uses_host_environment(tmp_path: Path) -> None:
    repo = tmp_path
    pool = repo / "persona" / "datasets" / "matraix-persona-dev-sample"
    pool.mkdir(parents=True)
    (pool / "persona_0001.yaml").write_text(
        "persona_id: '0001'\nversion: '1.0'\nsource: Nemotron\ndimensions: {}\n",
        encoding="utf-8",
    )

    job = build_application_job_config(
        {
            "name": "auto-survey",
            "task": "application/tasks/example-survey_product-feedback",
            "persona_pool": "persona/datasets/matraix-persona-dev-sample",
            "persona_ids": ["0001"],
            "execution_mode": "auto",
            "trial_profile": "json_survey",
            "agent": {"name": "persona-json-survey", "model_name": "anthropic/claude-haiku-4-5"},
            "job": {"job_name": "auto-survey", "jobs_dir": "jobs"},
        },
        repo_root=repo,
    )
    meta = job.pop("_job_meta")
    assert meta["trial_profile"] == "json_survey"
    assert job["environment"] == {"type": "host", "delete": True}
    assert job["tasks"][0]["path"] == "application/tasks/example-survey_product-feedback"
    assert resolve_harbor_task_path(
        "application/tasks/example-survey_product-feedback",
        trial_profile="json_survey",
    ) == "application/tasks/example-survey_product-feedback"


def test_build_application_job_config_use_entire_pool(tmp_path: Path) -> None:
    repo = tmp_path
    pool = repo / "persona" / "datasets" / "matraix-persona-dev-sample"
    pool.mkdir(parents=True)
    for pid in ("0001", "0002", "0003"):
        (pool / f"persona_{pid}.yaml").write_text(
            f"persona_id: '{pid}'\nversion: '1.0'\nsource: Nemotron\ndimensions: {{}}\n",
            encoding="utf-8",
        )

    job = build_application_job_config(
        {
            "name": "entire-pool-job",
            "task": "application/tasks/example-survey_product-feedback",
            "persona_pool": "persona/datasets/matraix-persona-dev-sample",
            "use_entire_pool": True,
            "sample_size": 1,
            "execution_mode": "auto",
            "trial_profile": "json_survey",
            "agent": {"name": "persona-claude-code", "model_name": "anthropic/claude-haiku-4-5"},
            "job": {"job_name": "entire-pool-job", "jobs_dir": "jobs"},
        },
        repo_root=repo,
    )
    meta = job.pop("_job_meta")
    assert meta["use_entire_pool"] is True
    assert sorted(meta["selected_persona_ids"]) == ["0001", "0002", "0003"]
    assert len(job["agents"]) == 3


def test_build_application_job_config_rejects_unknown_mode(tmp_path: Path) -> None:
    repo = tmp_path
    pool = repo / "persona" / "datasets" / "matraix-persona-dev-sample"
    pool.mkdir(parents=True)
    (pool / "persona_0001.yaml").write_text(
        "persona_id: '0001'\nversion: '1.0'\nsource: Nemotron\ndimensions: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="execution_mode"):
        build_application_job_config(
            {
                "name": "bad-mode",
                "task": "application/tasks/example-survey_product-feedback",
                "persona_pool": "persona/datasets/matraix-persona-dev-sample",
                "sample_size": 1,
                "execution_mode": "turbo",
                "agent": {"name": "persona-claude-code", "model_name": "anthropic/claude-haiku-4-5"},
            },
            repo_root=repo,
        )



def _condition_personas() -> list[dict]:
    return [
        {
            "persona_id": f"{index:04d}",
            "path": f"persona_{index:04d}.yaml",
            "dimensions": {"segment": segment},
        }
        for index, segment in enumerate(
            ["alpha"] * 5 + ["beta"] * 5,
            start=1,
        )
    ]


def _conditions() -> list[dict]:
    return [
        {"id": "a_first", "task": "application/tasks/survey_a_first"},
        {"id": "b_first", "task": "application/tasks/survey_b_first"},
    ]


def test_condition_assignment_is_deterministic_and_order_independent(
    tmp_path: Path,
) -> None:
    personas = _condition_personas()

    first = build_condition_assignment_manifest(
        personas,
        conditions=_conditions(),
        block_by=["segment"],
        seed=20260816,
        repo_root=tmp_path,
    )
    second = build_condition_assignment_manifest(
        list(reversed(personas)),
        conditions=list(reversed(_conditions())),
        block_by=["segment"],
        seed=20260816,
        repo_root=tmp_path,
    )

    assert first == second


def test_condition_assignment_balances_each_block(tmp_path: Path) -> None:
    manifest = build_condition_assignment_manifest(
        _condition_personas(),
        conditions=_conditions(),
        block_by=["segment"],
        seed=20260816,
        repo_root=tmp_path,
    )

    assert manifest["method"] == "blocked_randomization"
    assert manifest["seed"] == 20260816
    assert manifest["block_by"] == ["segment"]
    assert len(manifest["assignments"]) == 10

    by_segment: dict[str, dict[str, int]] = {}
    for assignment in manifest["assignments"]:
        segment = assignment["block"]["segment"]
        condition_id = assignment["condition_id"]
        by_segment.setdefault(segment, {})
        by_segment[segment][condition_id] = (
            by_segment[segment].get(condition_id, 0) + 1
        )

    assert set(by_segment) == {"alpha", "beta"}
    for counts in by_segment.values():
        assert set(counts) == {"a_first", "b_first"}
        assert max(counts.values()) - min(counts.values()) <= 1


def test_condition_assignment_assigns_every_persona_once(tmp_path: Path) -> None:
    personas = _condition_personas()

    manifest = build_condition_assignment_manifest(
        personas,
        conditions=_conditions(),
        block_by=["segment"],
        seed=42,
        repo_root=tmp_path,
    )

    assigned_ids = [
        assignment["persona_id"] for assignment in manifest["assignments"]
    ]

    assert len(assigned_ids) == len(personas)
    assert len(set(assigned_ids)) == len(personas)
    assert set(assigned_ids) == {
        str(persona["persona_id"]) for persona in personas
    }


def test_condition_assignment_rejects_missing_block_field(tmp_path: Path) -> None:
    personas = _condition_personas()
    personas[0] = {
        "persona_id": personas[0]["persona_id"],
        "path": personas[0]["path"],
        "dimensions": {},
    }

    with pytest.raises(ValueError, match="segment"):
        build_condition_assignment_manifest(
            personas,
            conditions=_conditions(),
            block_by=["segment"],
            seed=42,
            repo_root=tmp_path,
        )


def test_condition_assignment_rejects_duplicate_condition_ids(
    tmp_path: Path,
) -> None:
    conditions = [
        {"id": "same", "task": "application/tasks/survey_a"},
        {"id": "same", "task": "application/tasks/survey_b"},
    ]

    with pytest.raises(ValueError, match="condition"):
        build_condition_assignment_manifest(
            _condition_personas(),
            conditions=conditions,
            block_by=["segment"],
            seed=42,
            repo_root=tmp_path,
        )


def _write_condition_pool(repo: Path) -> str:
    relative_pool = "persona/datasets/condition-test"
    pool = repo / relative_pool
    pool.mkdir(parents=True)
    for index, segment in enumerate(["alpha"] * 3 + ["beta"] * 3, start=1):
        (pool / f"persona_{index:04d}.yaml").write_text(
            f"persona_id: '{index:04d}'\ndimensions:\n  segment: {segment}\n",
            encoding="utf-8",
        )
    return relative_pool


def _condition_job_spec(persona_pool: str) -> dict:
    return {
        "name": "condition-job",
        "persona_pool": persona_pool,
        "sample_size": 6,
        "seed": 42,
        "execution_mode": "auto",
        "trial_profile": "json_survey",
        "agent": {"name": "persona-json-survey", "model_name": "anthropic/claude-haiku-4-5"},
        "conditions": list(reversed(_conditions())),
        "assignment": {
            "method": "blocked_randomization",
            "block_by": ["segment"],
            "seed": 20260816,
        },
    }


def test_condition_job_compiles_manifest_into_exact_harbor_pairs(tmp_path: Path) -> None:
    persona_pool = _write_condition_pool(tmp_path)
    job = build_application_job_config(
        _condition_job_spec(persona_pool), repo_root=tmp_path
    )
    meta = job["_job_meta"]
    manifest = meta["condition_assignment"]

    assert manifest["method"] == "blocked_randomization"
    assert manifest["seed"] == 20260816
    assert manifest["block_by"] == ["segment"]
    assert manifest["conditions"] == _conditions()
    assert job["tasks"] == [{"path": condition["task"]} for condition in _conditions()]
    assert len(job["agents"]) == len(job["trial_plan"]) == 6
    assert len(manifest["assignments"]) == 6
    assert [item["persona_id"] for item in manifest["assignments"]] == sorted(
        meta["selected_persona_ids"]
    )
    assert len({item["persona_id"] for item in manifest["assignments"]}) == 6
    assert sorted(item["agent_index"] for item in job["trial_plan"]) == list(range(6))
    selected = resolve_persona_entries(
        meta["selected_persona_ids"], persona_pool=persona_pool, repo_root=tmp_path
    )
    assert manifest == build_condition_assignment_manifest(
        selected,
        conditions=list(reversed(_conditions())),
        block_by=["segment"],
        seed=20260816,
        repo_root=tmp_path,
    )

    condition_tasks = {item["id"]: item["task"] for item in manifest["conditions"]}
    for plan_entry, assignment in zip(job["trial_plan"], manifest["assignments"]):
        task = job["tasks"][plan_entry["task_index"]]["path"]
        agent_kwargs = job["agents"][plan_entry["agent_index"]]["kwargs"]
        assert agent_kwargs["persona_path"] == assignment["persona_path"]
        assert task == assignment["task"] == condition_tasks[assignment["condition_id"]]
        assert agent_kwargs["survey_task_path"] == task
        assert assignment["block"]["segment"] in {"alpha", "beta"}

    harbor_config = JobConfig.model_validate(
        {key: value for key, value in job.items() if key != "_job_meta"}
    )
    assert len(harbor_config.trial_plan) == 6
    assert meta["seed"] == 42


def test_condition_job_defaults_assignment_seed_to_sampling_seed(tmp_path: Path) -> None:
    spec = _condition_job_spec(_write_condition_pool(tmp_path))
    del spec["assignment"]["seed"]

    job = build_application_job_config(spec, repo_root=tmp_path)

    assert job["_job_meta"]["seed"] == 42
    assert job["_job_meta"]["condition_assignment"]["seed"] == 42


def test_condition_job_rejects_unsupported_assignment_method(tmp_path: Path) -> None:
    spec = _condition_job_spec(_write_condition_pool(tmp_path))
    spec["assignment"]["method"] = "round_robin_magic"

    with pytest.raises(ValueError, match="assignment method"):
        build_application_job_config(spec, repo_root=tmp_path)


def test_condition_job_rejects_non_mapping_assignment(tmp_path: Path) -> None:
    spec = _condition_job_spec(_write_condition_pool(tmp_path))
    spec["assignment"] = []

    with pytest.raises(ValueError, match="assignment must be a mapping"):
        build_application_job_config(spec, repo_root=tmp_path)


def test_condition_job_requires_json_survey(tmp_path: Path) -> None:
    spec = _condition_job_spec(_write_condition_pool(tmp_path))
    spec["trial_profile"] = "user_sim_chat"

    with pytest.raises(ValueError, match="json_survey"):
        build_application_job_config(spec, repo_root=tmp_path)


def test_condition_job_rejects_empty_conditions(tmp_path: Path) -> None:
    spec = _condition_job_spec(_write_condition_pool(tmp_path))
    spec["conditions"] = []

    with pytest.raises(ValueError, match="condition"):
        build_application_job_config(spec, repo_root=tmp_path)


def test_job_without_conditions_preserves_single_task_and_agent_shape(
    tmp_path: Path,
) -> None:
    persona_pool = _write_condition_pool(tmp_path)
    spec = _condition_job_spec(persona_pool)
    del spec["conditions"]
    del spec["assignment"]
    spec["task"] = "application/tasks/example-survey_product-feedback"
    spec["persona_ids"] = ["0003", "0001"]

    job = build_application_job_config(spec, repo_root=tmp_path)

    assert job["tasks"] == [{"path": spec["task"]}]
    assert "trial_plan" not in job
    assert "condition_assignment" not in job["_job_meta"]
    assert job["_job_meta"]["selected_persona_ids"] == ["0003", "0001"]
    assert [Path(agent["kwargs"]["persona_path"]).stem for agent in job["agents"]] == [
        "persona_0003",
        "persona_0001",
    ]
    assert all("survey_task_path" not in agent["kwargs"] for agent in job["agents"])
