"""Trial planning for Playground's local distributed Harbor runner."""

from __future__ import annotations

import json
import shutil

import pytest

from backend.service.local_distributed_harbor import LocalDistributedHarborCoordinator


def _coordinator(tmp_path, job_config):
    return LocalDistributedHarborCoordinator(
        repo_root=tmp_path,
        job_name="planned-job",
        job_config=job_config,
        launch_env={},
        command_runner=lambda *args, **kwargs: 0,
        harbor_command=("harbor",),
    )


def _pairings(configs):
    return [
        (config["task"]["path"], config["agent"]["kwargs"]["persona_path"])
        for config in configs
    ]


def test_explicit_plan_preserves_pairings_and_attempt_order(tmp_path):
    config = {
        "tasks": [{"path": "task-a"}, {"path": "task-b"}],
        "agents": [
            {"kwargs": {"persona_path": "persona-1"}},
            {"kwargs": {"persona_path": "persona-2"}},
        ],
        "trial_plan": [
            {"task_index": 1, "agent_index": 0},
            {"task_index": 0, "agent_index": 1},
        ],
        "n_attempts": 2,
        "verifier": {"disable": True},
    }
    planned = _coordinator(tmp_path, config)._planned_trial_configs()

    assert _pairings(planned) == [
        ("task-b", "persona-1"),
        ("task-a", "persona-2"),
        ("task-b", "persona-1"),
        ("task-a", "persona-2"),
    ]
    assert all(row["verifier"] == config["verifier"] for row in planned)
    assert len({row["trial_name"] for row in planned}) == 4


@pytest.mark.parametrize("plan", [None, []])
def test_absent_or_empty_plan_preserves_cartesian_order(tmp_path, plan):
    config = {
        "tasks": [{"path": "task-a"}, {"path": "task-b"}],
        "agents": [
            {"kwargs": {"persona_path": "persona-1"}},
            {"kwargs": {"persona_path": "persona-2"}},
        ],
        "n_attempts": 2,
    }
    if plan is not None:
        config["trial_plan"] = plan
    planned = _coordinator(tmp_path, config)._planned_trial_configs()

    assert _pairings(planned) == [
        ("task-a", "persona-1"),
        ("task-a", "persona-2"),
        ("task-b", "persona-1"),
        ("task-b", "persona-2"),
    ] * 2


@pytest.mark.parametrize(
    ("entry", "field"),
    [
        ({"task_index": 2, "agent_index": 0}, "task_index"),
        ({"task_index": 0, "agent_index": 2}, "agent_index"),
    ],
)
def test_explicit_plan_rejects_out_of_range_indices(tmp_path, entry, field):
    coordinator = _coordinator(
        tmp_path,
        {
            "tasks": [{"path": "task-a"}],
            "agents": [{"kwargs": {"persona_path": "persona-1"}}],
            "trial_plan": [entry],
        },
    )
    with pytest.raises(ValueError, match=field):
        coordinator._planned_trial_configs()


def test_planned_retry_reuses_trial_names_and_runs_only_missing_attempt(tmp_path):
    config = {
        "tasks": [{"path": "task-a"}, {"path": "task-b"}],
        "agents": [
            {"kwargs": {"persona_path": "persona-1"}},
            {"kwargs": {"persona_path": "persona-2"}},
        ],
        "trial_plan": [
            {"task_index": 1, "agent_index": 0},
            {"task_index": 0, "agent_index": 1},
        ],
        "n_attempts": 2,
        "n_concurrent_trials": 1,
    }
    first = _coordinator(tmp_path, config)
    invoked: list[str] = []

    def first_worker(manifest_path, _env):
        trial = json.loads(manifest_path.read_text())
        invoked.append(trial["trial_name"])
        result = {"trial_name": trial["trial_name"]}
        if len(invoked) == 3:
            result["exception_info"] = {"exception_type": "WorkerError"}
        result_path = first.job_dir / trial["trial_name"] / "result.json"
        result_path.write_text(json.dumps(result))
        return 0

    first.worker_runner = first_worker
    assert first.run()["trialCount"] == 4
    names = json.loads(first.state_path.read_text())["trialNames"]
    assert names == invoked
    assert len(set(names)) == 4
    manifests = [
        json.loads((first.manifests_dir / f"{name}.json").read_text())
        for name in names
    ]
    assert _pairings(manifests) == [
        ("task-b", "persona-1"),
        ("task-a", "persona-2"),
    ] * 2
    successful_results = {
        name: (first.job_dir / name / "result.json").read_bytes()
        for name in names
        if name != names[2]
    }
    original_job_id = first.job_id
    shutil.rmtree(first.job_dir / names[2])

    resumed = _coordinator(tmp_path, config)
    retried: list[str] = []

    def retry_worker(manifest_path, _env):
        trial = json.loads(manifest_path.read_text())
        retried.append(trial["trial_name"])
        result_path = resumed.job_dir / trial["trial_name"] / "result.json"
        result_path.write_text(json.dumps({"trial_name": trial["trial_name"]}))
        return 0

    resumed.worker_runner = retry_worker
    assert resumed.run()["trialCount"] == 4
    assert retried == [names[2]]
    assert json.loads(resumed.state_path.read_text())["trialNames"] == names
    assert resumed.job_id == original_job_id
    assert json.loads((resumed.job_dir / "result.json").read_text())["stats"]["n_completed_trials"] == 4
    assert {
        name: (resumed.job_dir / name / "result.json").read_bytes()
        for name in successful_results
    } == successful_results
