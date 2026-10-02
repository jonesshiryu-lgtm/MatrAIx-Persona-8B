"""Execute Harbor jobs on a remote runner worker."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

_ALLOWED_REMOTE_HARBOR_ENV_KEYS = frozenset({"PYTHONPATH"})


def is_allowed_remote_harbor_env_key(key: str) -> bool:
    """Return True when ``key`` may cross the Remote Runner HTTP boundary."""
    return key in _ALLOWED_REMOTE_HARBOR_ENV_KEYS or key.startswith("MATRIX_")


def filter_remote_harbor_payload_env(env: dict[str, Any]) -> dict[str, str]:
    """Drop secrets and unrelated process env from a remote ``harbor_job`` payload."""
    return {
        str(key): str(value)
        for key, value in env.items()
        if is_allowed_remote_harbor_env_key(str(key))
    }


def resolve_repo_root(payload: dict[str, Any]) -> Path:
    raw = str(payload.get("repoRoot") or "").strip()
    if raw:
        return Path(raw).resolve()
    return Path.cwd().resolve()


def build_harbor_env(*, repo_root: Path, payload: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    extra = payload.get("env")
    if isinstance(extra, dict):
        for key, value in filter_remote_harbor_payload_env(extra).items():
            env[key] = value
    existing = env.get("PYTHONPATH", "")
    path_entries = [entry for entry in existing.split(":") if entry]
    required_paths = [
        str(repo_root),
        str(repo_root / "src"),
        str(repo_root / "environment" / "runtime"),
        str(repo_root / "environment" / "agents"),
        str(repo_root / "packages" / "playground" / "src"),
        str(repo_root / "application" / "playground"),
    ]
    for path in reversed(required_paths):
        if path not in path_entries:
            path_entries.insert(0, path)
    env["PYTHONPATH"] = ":".join(path_entries)
    return env


def default_harbor_command() -> list[str]:
    override = os.environ.get("REMOTE_RUNNER_HARBOR_COMMAND", "").strip()
    if override:
        return shlex.split(override)
    from playground.harbor.playground import (
        _default_harbor_command,
    )

    return list(_default_harbor_command())


def write_incoming_job_config(
    *,
    repo_root: Path,
    job_name: str,
    config_yaml: str,
    runs_dir: Path | None = None,
) -> Path:
    base = runs_dir or (repo_root / "configs" / "jobs" / "remote-runner-incoming")
    base.mkdir(parents=True, exist_ok=True)
    safe_name = job_name.replace("/", "_").replace("\\", "_") or "remote-job"
    config_path = base / "{}.yaml".format(safe_name)
    config_path.write_text(config_yaml, encoding="utf-8")
    return config_path


def rel_config_path(config_path: Path, repo_root: Path) -> str:
    try:
        return str(config_path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(config_path.resolve())


def run_harbor_job(
    payload: dict[str, Any],
    *,
    command_runner: Callable[..., int] | None = None,
    harbor_command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Materialize a Harbor job config and run ``harbor run`` on this worker."""
    config_yaml = str(payload.get("configYaml") or "").strip()
    if not config_yaml:
        raise ValueError("harbor_job payload requires configYaml")

    repo_root = resolve_repo_root(payload)
    job_name = str(payload.get("jobName") or "remote-job").strip() or "remote-job"
    config_path = write_incoming_job_config(
        repo_root=repo_root,
        job_name=job_name,
        config_yaml=config_yaml,
    )
    command = list(harbor_command or default_harbor_command()) + [
        "--yes",
        "-c",
        rel_config_path(config_path, repo_root),
    ]
    env = build_harbor_env(repo_root=repo_root, payload=payload)

    if command_runner is not None:
        exit_code = command_runner(command, cwd=repo_root, env=env)
    else:
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            env=env,
            check=False,
        )
        exit_code = int(completed.returncode)

    jobs_dir = str(payload.get("jobsDir") or "jobs")
    result = {
        "jobName": job_name,
        "configPath": rel_config_path(config_path, repo_root),
        "jobsDir": jobs_dir,
        "exitCode": exit_code,
    }
    if exit_code != 0:
        raise RuntimeError("harbor run exited with code {}".format(exit_code))
    return result


_SKIP_DIR_NAMES = frozenset(
    {
        "_generated",
        "agent",
        "artifacts",
        "verifier",
        "logs",
        "__pycache__",
    }
)


def resolve_job_dir(payload: dict[str, Any], *, repo_root: Path | None = None) -> Path:
    """Resolve the directory holding trial runs for a Harbor job."""
    if repo_root is None:
        repo_root = resolve_repo_root(payload)
    if payload.get("jobDir"):
        raw_job_dir = Path(str(payload["jobDir"]))
        return raw_job_dir if raw_job_dir.is_absolute() else (repo_root / raw_job_dir)
    jobs_dir_val = str(payload.get("jobsDir") or "jobs").strip() or "jobs"
    jobs_path = Path(jobs_dir_val)
    if not jobs_path.is_absolute():
        jobs_path = repo_root / jobs_path
    job_name = str(payload.get("jobName") or "remote-job").strip() or "remote-job"
    return jobs_path / job_name


def discover_harbor_job_artifacts(payload: dict[str, Any]) -> dict[str, Any]:
    """Discover trial JSON task outputs under `artifacts/app/output/`.

    Registers:
    - Canonical relative path from job_dir (e.g. `<trial>/artifacts/app/output/<name>.json`)
    - Trial-scoped shortcut (e.g. `<trial>/<name>.json`)
    - Trial-relative path if only a single trial exists (e.g. `artifacts/app/output/<name>.json`)
    - Basename shortcut only when the filename is unique across all trials
    - `artifact_manifest.json` listing the canonical relative paths
    """
    repo_root = resolve_repo_root(payload)
    job_dir = resolve_job_dir(payload, repo_root=repo_root)
    job_name = str(payload.get("jobName") or "remote-job").strip() or "remote-job"

    if not job_dir.is_dir():
        manifest = {
            "jobName": job_name,
            "artifacts": [],
            "entries": [],
            "trials": {},
        }
        return {"artifact_manifest.json": manifest}

    trial_dirs: list[Path] = []
    for child in sorted(job_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name.startswith("_") or child.name in _SKIP_DIR_NAMES:
            continue
        trial_dirs.append(child)

    # Support flat single-trial layout where artifacts/app/output is placed directly in job_dir
    if not trial_dirs and (job_dir / "artifacts" / "app" / "output").is_dir():
        trial_dirs.append(job_dir)

    discovered: list[tuple[str, Path, Path, str, Any]] = []
    for t_dir in trial_dirs:
        output_dir = t_dir / "artifacts" / "app" / "output"
        if not output_dir.is_dir():
            for fallback in (t_dir / "artifacts" / "output", t_dir / "output"):
                if fallback.is_dir():
                    output_dir = fallback
                    break
        if not output_dir.is_dir():
            continue

        trial_name = t_dir.name if t_dir != job_dir else "default"
        for json_path in sorted(p for p in output_dir.rglob("*.json") if p.is_file()):
            try:
                content = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            try:
                rel_to_job = str(json_path.relative_to(job_dir))
            except ValueError:
                rel_to_job = f"{trial_name}/artifacts/app/output/{json_path.name}"
            discovered.append((trial_name, t_dir, json_path, rel_to_job, content))

    basename_counts: dict[str, int] = {}
    for _, _, json_path, _, _ in discovered:
        basename_counts[json_path.name] = basename_counts.get(json_path.name, 0) + 1

    artifacts: dict[str, Any] = {}

    # 1. Register canonical relative path from job_dir
    for _, _, _, rel_to_job, content in discovered:
        artifacts[rel_to_job] = content

    # 2. Register trial-scoped shortcut: <trial_name>/<filename>
    for trial_name, _, json_path, _, content in discovered:
        trial_shortcut = f"{trial_name}/{json_path.name}"
        if trial_shortcut not in artifacts:
            artifacts[trial_shortcut] = content

    # 3. If single trial, register trial-relative path (artifacts/app/output/<name>.json)
    if len(trial_dirs) == 1:
        for _, t_dir, json_path, _, content in discovered:
            try:
                rel_to_trial = str(json_path.relative_to(t_dir))
                artifacts[rel_to_trial] = content
            except ValueError:
                pass

    # 4. Expose basename shortcut ONLY when unique across all trials
    for _, _, json_path, _, content in discovered:
        if basename_counts[json_path.name] == 1:
            artifacts[json_path.name] = content

    # 5. Build artifact_manifest.json
    canonical_names = [item[3] for item in discovered]
    trial_entries: dict[str, list[str]] = {}
    for trial_name, _, _, rel_to_job, _ in discovered:
        trial_entries.setdefault(trial_name, []).append(rel_to_job)

    manifest = {
        "jobName": job_name,
        "artifacts": canonical_names,
        "entries": canonical_names,
        "trials": trial_entries,
    }
    artifacts["artifact_manifest.json"] = manifest

    return artifacts

