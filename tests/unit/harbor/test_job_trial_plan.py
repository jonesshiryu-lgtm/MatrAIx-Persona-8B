from pathlib import Path

import pytest

from harbor.job import Job
from harbor.models.job.config import JobConfig, TrialPlanEntry
from harbor.models.job.lock import build_job_lock
from harbor.models.trial.config import AgentConfig, TaskConfig


def _task(name: str, digest_char: str) -> TaskConfig:
    return TaskConfig(
        name=f"test-org/{name}",
        ref=f"sha256:{digest_char * 64}",
    )


def _agent(persona_id: str) -> AgentConfig:
    return AgentConfig(
        name="oracle",
        kwargs={"persona_id": persona_id},
    )


def _job(
    tmp_path: Path,
    *,
    tasks: list[TaskConfig],
    agents: list[AgentConfig],
    trial_plan: list[TrialPlanEntry] | None = None,
    n_attempts: int = 1,
) -> Job:
    plan_kwargs = {"trial_plan": trial_plan} if trial_plan is not None else {}
    config = JobConfig(
        job_name="trial-plan-test",
        jobs_dir=tmp_path,
        tasks=tasks,
        agents=agents,
        n_attempts=n_attempts,
        **plan_kwargs,
    )
    return Job(
        config,
        _task_configs=tasks,
        _metrics={"adhoc": []},
    )


@pytest.mark.parametrize("trial_plan", [None, []])
def test_job_without_trial_plan_keeps_cartesian_product(
    tmp_path: Path, trial_plan: list[TrialPlanEntry] | None
) -> None:
    tasks = [
        _task("condition-a", "a"),
        _task("condition-b", "b"),
    ]
    agents = [
        _agent("0001"),
        _agent("0002"),
    ]

    job = _job(
        tmp_path,
        tasks=tasks,
        agents=agents,
        trial_plan=trial_plan,
    )
    try:
        assert [(trial.task, trial.agent) for trial in job._trial_configs] == [
            (tasks[0], agents[0]),
            (tasks[0], agents[1]),
            (tasks[1], agents[0]),
            (tasks[1], agents[1]),
        ]
    finally:
        job._close_logger_handlers()


def test_trial_plan_creates_only_explicit_task_agent_pairs(tmp_path: Path) -> None:
    tasks = [
        _task("condition-a", "a"),
        _task("condition-b", "b"),
    ]
    agents = [
        _agent("0001"),
        _agent("0002"),
    ]

    job = _job(
        tmp_path,
        tasks=tasks,
        agents=agents,
        trial_plan=[
            TrialPlanEntry(task_index=0, agent_index=0),
            TrialPlanEntry(task_index=1, agent_index=1),
        ],
    )
    try:
        assert len(job._trial_configs) == 2

        assert job._trial_configs[0].task == tasks[0]
        assert job._trial_configs[0].agent == agents[0]

        assert job._trial_configs[1].task == tasks[1]
        assert job._trial_configs[1].agent == agents[1]
    finally:
        job._close_logger_handlers()


def test_trial_plan_preserves_n_attempts(tmp_path: Path) -> None:
    tasks = [
        _task("condition-a", "a"),
        _task("condition-b", "b"),
    ]
    agents = [
        _agent("0001"),
        _agent("0002"),
    ]

    job = _job(
        tmp_path,
        tasks=tasks,
        agents=agents,
        trial_plan=[
            TrialPlanEntry(task_index=0, agent_index=0),
            TrialPlanEntry(task_index=1, agent_index=1),
        ],
        n_attempts=2,
    )
    try:
        assert [(trial.task, trial.agent) for trial in job._trial_configs] == [
            (tasks[0], agents[0]),
            (tasks[1], agents[1]),
        ] * 2
    finally:
        job._close_logger_handlers()


def test_trial_plan_lock_tracks_task_agent_pairings(tmp_path: Path) -> None:
    tasks = [_task("condition-a", "a"), _task("condition-b", "b")]
    agents = [_agent("0001"), _agent("0002")]
    first = _job(
        tmp_path / "first",
        tasks=tasks,
        agents=agents,
        trial_plan=[
            TrialPlanEntry(task_index=0, agent_index=0),
            TrialPlanEntry(task_index=1, agent_index=1),
        ],
    )
    second = _job(
        tmp_path / "second",
        tasks=tasks,
        agents=agents,
        trial_plan=[
            TrialPlanEntry(task_index=0, agent_index=1),
            TrialPlanEntry(task_index=1, agent_index=0),
        ],
    )
    try:
        first_lock = build_job_lock(
            config=first.config,
            trial_configs=first._trial_configs,
            invocation=["harbor", "run"],
        )
        second_lock = build_job_lock(
            config=second.config,
            trial_configs=second._trial_configs,
            invocation=["harbor", "run"],
        )

        assert [
            (trial.task.digest, trial.agent.kwargs["persona_id"])
            for trial in first_lock.trials
        ] == [("sha256:" + "a" * 64, "0001"), ("sha256:" + "b" * 64, "0002")]
        assert [
            (trial.task.digest, trial.agent.kwargs["persona_id"])
            for trial in second_lock.trials
        ] == [("sha256:" + "a" * 64, "0002"), ("sha256:" + "b" * 64, "0001")]
        assert first_lock != second_lock
        assert "trial_plan" not in first_lock.model_dump(mode="json")
    finally:
        first._close_logger_handlers()
        second._close_logger_handlers()


def test_trial_plan_rejects_out_of_range_task_index(tmp_path: Path) -> None:
    tasks = [_task("condition-a", "a")]
    agents = [_agent("0001")]

    with pytest.raises(ValueError, match="task_index"):
        _job(
            tmp_path,
            tasks=tasks,
            agents=agents,
            trial_plan=[
                TrialPlanEntry(task_index=1, agent_index=0),
            ],
        )


def test_trial_plan_rejects_out_of_range_agent_index(tmp_path: Path) -> None:
    tasks = [_task("condition-a", "a")]
    agents = [_agent("0001")]

    with pytest.raises(ValueError, match="agent_index"):
        _job(
            tmp_path,
            tasks=tasks,
            agents=agents,
            trial_plan=[
                TrialPlanEntry(task_index=0, agent_index=1),
            ],
        )
