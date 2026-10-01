"""Tests for application job generation helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from harbor.models.job.config import JobConfig
from matraix.application_job import collect_run_env_exports
from matraix.compute_family import ComputePlan
from matraix.provider_credentials import export_hint_lines, resolve_provider_credential

REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = REPO_ROOT / "application" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import generate_application_job as gen  # noqa: E402


def test_collect_run_env_exports_survey() -> None:
    exports = collect_run_env_exports(
        trial_profile="json_survey",
        task_path="application/tasks/example-survey_product-feedback",
        repo_root=REPO_ROOT,
    )
    assert exports == [("MATRIX_SURVEY_TASK_PATH", "application/tasks/example-survey_product-feedback")]


def test_collect_run_env_exports_chat() -> None:
    exports = collect_run_env_exports(
        trial_profile="user_sim_chat",
        task_path="application/tasks/chat_meal-planning-nutrition",
        repo_root=REPO_ROOT,
    )
    assert exports == [
        ("MATRIX_CHATBOT_TASK_PATH", "application/tasks/chat_meal-planning-nutrition")
    ]


def test_export_hint_lines_are_provider_aware() -> None:
    assert export_hint_lines("openai/gpt-4o-mini") == ["export OPENAI_API_KEY=..."]
    assert export_hint_lines("dashscope/deepseek-v3.2") == [
        "export DASHSCOPE_API_KEY=..."
    ]
    assert resolve_provider_credential("openai/gpt-4o-mini").env_var != "ANTHROPIC_API_KEY"


def test_format_run_instructions_local_uses_matraix_run() -> None:
    lines = gen.format_run_instructions(
        config_display="configs/jobs/application-task-job-recipe/job.yaml",
        compute_plan=ComputePlan(family="local", environment="host"),
        n_concurrent_trials=2,
    )
    joined = "\n".join(lines)
    assert "uv run matraix run -c" in joined
    assert "--launch" not in joined
    assert "HarborJobService" not in joined


def test_format_run_instructions_modal_uses_matraix_run() -> None:
    lines = gen.format_run_instructions(
        config_display="configs/jobs/application-task-job-recipe/job.yaml",
        compute_plan=ComputePlan(
            family="modal", environment="host", dispatch="modal_jobs"
        ),
        n_concurrent_trials=32,
    )
    joined = "\n".join(lines)
    assert "uv run matraix run -c configs/jobs/application-task-job-recipe/job.yaml" in joined
    assert "--launch" not in joined
    assert "sidecar computeFamily=modal" in joined
    assert "generate_application_job.py" not in joined


def _generate_job(tmp_path: Path, monkeypatch, *extra_args: str) -> tuple[dict, dict]:
    pool = tmp_path / "persona" / "datasets" / "condition-test"
    pool.mkdir(parents=True)
    for index, segment in enumerate(("alpha", "alpha", "beta", "beta"), start=1):
        (pool / f"persona_{index:04d}.yaml").write_text(
            f"persona_id: '{index:04d}'\nsource: test\ndimensions:\n  segment: {segment}\n",
            encoding="utf-8",
        )
    for name in ("base_survey", "condition_a", "condition_b"):
        task = tmp_path / "application" / "tasks" / name
        task.mkdir(parents=True)
        (task / "task.toml").write_text('[metadata]\ntype = "survey"\n', encoding="utf-8")

    out_path = tmp_path / "generated.yaml"
    monkeypatch.setattr(gen, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_application_job.py",
            "--task",
            "application/tasks/base_survey",
            "--no-strategy",
            "--dataset",
            "persona/datasets/condition-test",
            "--persona-ids",
            "0001",
            "0002",
            "0003",
            "0004",
            "--seed",
            "11",
            "--name",
            "condition-test",
            "--compute-family",
            "local",
            "--out",
            str(out_path),
            *extra_args,
        ],
    )
    gen.main()
    job = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    meta = json.loads(out_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    return job, meta


def test_generator_writes_condition_plan_and_assignment_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    job, meta = _generate_job(
        tmp_path,
        monkeypatch,
        "--condition",
        "b=application/tasks/condition_b",
        "--condition",
        "a=application/tasks/condition_a",
        "--block-by",
        "segment,source",
        "--block-by",
        "segment",
        "--assignment-seed",
        "29",
    )

    manifest = meta["condition_assignment"]
    assert meta["task"] == "application/tasks/base_survey"
    assert meta["seed"] == 11
    assert manifest["seed"] == 29
    assert manifest["block_by"] == ["segment", "source"]
    assert manifest["conditions"] == [
        {"id": "a", "task": "application/tasks/condition_a"},
        {"id": "b", "task": "application/tasks/condition_b"},
    ]
    assert job["tasks"] == [{"path": row["task"]} for row in manifest["conditions"]]
    assert "datasets" not in job
    assert len(job["agents"]) == len(job["trial_plan"]) == len(manifest["assignments"]) == 4
    assert len(JobConfig.model_validate(job).trial_plan) == 4
    for pair, assignment in zip(job["trial_plan"], manifest["assignments"]):
        task = job["tasks"][pair["task_index"]]["path"]
        kwargs = job["agents"][pair["agent_index"]]["kwargs"]
        assert task == assignment["task"] == kwargs["survey_task_path"]
        assert kwargs["persona_path"] == assignment["persona_path"]


def test_generator_defaults_assignment_seed_to_sampling_seed(
    tmp_path: Path, monkeypatch
) -> None:
    _, meta = _generate_job(
        tmp_path,
        monkeypatch,
        "--condition",
        "a=application/tasks/condition_a",
        "--condition",
        "b=application/tasks/condition_b",
    )

    assert meta["seed"] == meta["condition_assignment"]["seed"] == 11
    assert meta["condition_assignment"]["block_by"] == []


def test_generator_without_conditions_keeps_single_task_recipe(
    tmp_path: Path, monkeypatch
) -> None:
    job, meta = _generate_job(tmp_path, monkeypatch)

    assert job["tasks"] == [{"path": "application/tasks/base_survey"}]
    assert "trial_plan" not in job
    assert "condition_assignment" not in meta
    assert all("survey_task_path" not in agent["kwargs"] for agent in job["agents"])
    assert meta["task"] == "application/tasks/base_survey"


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (("--condition", "missing-separator"), "expected ID=TASK"),
        (("--condition", "=application/tasks/condition_a"), "expected ID=TASK"),
        (("--condition", "a="), "expected ID=TASK"),
        (
            (
                "--condition",
                "a=application/tasks/condition_a",
                "--condition",
                "a=application/tasks/condition_b",
            ),
            "duplicate --condition id",
        ),
        (("--block-by", "segment"), "require --condition"),
        (("--assignment-seed", "29"), "require --condition"),
    ],
)
def test_generator_rejects_invalid_condition_flags(
    monkeypatch, capsys, extra_args: tuple[str, ...], message: str
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_application_job.py", "--task", "application/tasks/base_survey", *extra_args],
    )

    with pytest.raises(SystemExit) as exc_info:
        gen.main()

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err
