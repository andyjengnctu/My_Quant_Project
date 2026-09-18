"""Windows Task Scheduler deployment owner for Trading Market Data auto update.

The OS scheduler is deliberately dumb: it only wakes the canonical one-shot app.
All dataset timing, freshness, retry, quota and market-date discovery semantics
remain owned by the Market Data runtime services.
"""
from __future__ import annotations

import json
import ntpath
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, Mapping

from core.market_data_auto_update_policy import get_market_data_auto_update_policy

SCHEDULER_STATUS_UNSUPPORTED = "UNSUPPORTED"
SCHEDULER_STATUS_NOT_INSTALLED = "NOT_INSTALLED"
SCHEDULER_STATUS_INSTALLED_ENABLED = "INSTALLED_ENABLED"
SCHEDULER_STATUS_INSTALLED_DISABLED = "INSTALLED_DISABLED"
SCHEDULER_STATUS_DRIFTED_ENABLED = "DRIFTED_ENABLED"
SCHEDULER_STATUS_DRIFTED_DISABLED = "DRIFTED_DISABLED"
SCHEDULER_STATUS_ERROR = "ERROR"

PowerShellRunner = Callable[[str, Mapping[str, str]], tuple[int, str, str]]


_STATUS_SCRIPT = r'''$ErrorActionPreference = 'Stop'
$task = Get-ScheduledTask -TaskName $env:MQP_TASK_NAME -TaskPath '\' -ErrorAction SilentlyContinue
if ($null -eq $task) {
    [pscustomobject]@{ installed = $false } | ConvertTo-Json -Compress
    exit 0
}
$info = Get-ScheduledTaskInfo -TaskName $env:MQP_TASK_NAME -TaskPath '\'
$repeatMinutes = $null
$logonTrigger = $false
foreach ($trigger in @($task.Triggers)) {
    if ($trigger.CimClass.CimClassName -eq 'MSFT_TaskLogonTrigger') { $logonTrigger = $true }
    if ($null -ne $trigger.Repetition -and $null -ne $trigger.Repetition.Interval -and [string]$trigger.Repetition.Interval) {
        $intervalText = [string]$trigger.Repetition.Interval
        try {
            $repeatMinutes = [int][Math]::Round(([System.Xml.XmlConvert]::ToTimeSpan($intervalText)).TotalMinutes)
        } catch {
            try { $repeatMinutes = [int][Math]::Round(([TimeSpan]::Parse($intervalText)).TotalMinutes) } catch {}
        }
    }
}
$action = @($task.Actions)[0]
[pscustomobject]@{
    installed = $true
    state = [string]$task.State
    enabled = ([string]$task.State -ne 'Disabled')
    execute = [string]$action.Execute
    arguments = [string]$action.Arguments
    working_directory = [string]$action.WorkingDirectory
    interval_minutes = $repeatMinutes
    logon_trigger = $logonTrigger
    next_run_at = $(if ($info.NextRunTime.Year -ge 2000) { $info.NextRunTime.ToString('o') } else { $null })
    last_run_at = $(if ($info.LastRunTime.Year -ge 2000) { $info.LastRunTime.ToString('o') } else { $null })
    last_task_result = [int64]$info.LastTaskResult
    missed_runs = [int64]$info.NumberOfMissedRuns
} | ConvertTo-Json -Compress'''

_INSTALL_SCRIPT = r'''$ErrorActionPreference = 'Stop'
$wake = [int]$env:MQP_WAKE_MINUTES
$delay = [int]$env:MQP_INITIAL_DELAY_MINUTES
$limit = [int]$env:MQP_EXECUTION_LIMIT_MINUTES
$userId = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute $env:MQP_TASK_EXECUTE -Argument $env:MQP_TASK_ARGUMENTS -WorkingDirectory $env:MQP_WORKING_DIRECTORY
$repeatTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes($delay) -RepetitionInterval (New-TimeSpan -Minutes $wake)
$logonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $userId
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes $limit) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $env:MQP_TASK_NAME -TaskPath '\' -Description $env:MQP_TASK_DESCRIPTION -Action $action -Trigger @($repeatTrigger, $logonTrigger) -Settings $settings -Principal $principal -Force | Out-Null'''

_ENABLE_SCRIPT = r'''$ErrorActionPreference = 'Stop'
Enable-ScheduledTask -TaskName $env:MQP_TASK_NAME -TaskPath '\' | Out-Null'''

_DISABLE_SCRIPT = r'''$ErrorActionPreference = 'Stop'
Disable-ScheduledTask -TaskName $env:MQP_TASK_NAME -TaskPath '\' | Out-Null'''

_REMOVE_SCRIPT = r'''$ErrorActionPreference = 'Stop'
$task = Get-ScheduledTask -TaskName $env:MQP_TASK_NAME -TaskPath '\' -ErrorAction SilentlyContinue
if ($null -ne $task) { Unregister-ScheduledTask -TaskName $env:MQP_TASK_NAME -TaskPath '\' -Confirm:$false }'''


def _windowless_python_executable(python_executable: str) -> str:
    python_path = Path(python_executable)
    if python_path.name.lower() == "pythonw.exe":
        return str(python_path)
    return str(python_path.with_name("pythonw.exe"))


def _scheduled_action(*, project_root: Path, python_executable: str) -> tuple[str, str, str]:
    script = project_root / "apps" / "market_data_auto_update.py"
    execute = _windowless_python_executable(python_executable)
    arguments = subprocess.list2cmdline(
        [str(script), "--project-root", str(project_root), "--quiet"]
    )
    return execute, arguments, str(project_root)


def build_market_data_scheduler_spec(
    project_root: str | Path,
    *,
    python_executable: str | None = None,
) -> dict[str, object]:
    root = Path(project_root).resolve()
    policy = get_market_data_auto_update_policy()
    python_path = str(Path(python_executable or sys.executable).resolve())
    execute, arguments, working_directory = _scheduled_action(project_root=root, python_executable=python_path)
    return {
        "task_name": policy.scheduler_task_name,
        "wake_minutes": int(policy.scheduler_wake_minutes),
        "initial_delay_minutes": int(policy.scheduler_initial_delay_minutes),
        "execution_time_limit_minutes": int(policy.scheduler_execution_time_limit_minutes),
        "execute": execute,
        "arguments": arguments,
        "working_directory": working_directory,
        "app_path": "apps/market_data_auto_update.py",
        "python_executable": python_path,
        "logon_trigger": True,
    }


def _default_powershell_runner(script: str, env_overrides: Mapping[str, str]) -> tuple[int, str, str]:
    env = dict(os.environ)
    env.update({str(key): str(value) for key, value in env_overrides.items()})
    creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
            cwd=None,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
            creationflags=creationflags,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"
    return int(completed.returncode), str(completed.stdout or ""), str(completed.stderr or "")


def _invoke(script: str, env: Mapping[str, str], runner: PowerShellRunner | None) -> tuple[int, str, str]:
    return (runner or _default_powershell_runner)(script, env)


def _scheduler_env(spec: Mapping[str, object]) -> dict[str, str]:
    return {
        "MQP_TASK_NAME": str(spec["task_name"]),
        "MQP_TASK_EXECUTE": str(spec["execute"]),
        "MQP_TASK_ARGUMENTS": str(spec["arguments"]),
        "MQP_WORKING_DIRECTORY": str(spec["working_directory"]),
        "MQP_WAKE_MINUTES": str(int(spec["wake_minutes"])),
        "MQP_INITIAL_DELAY_MINUTES": str(int(spec["initial_delay_minutes"])),
        "MQP_EXECUTION_LIMIT_MINUTES": str(int(spec["execution_time_limit_minutes"])),
        "MQP_TASK_DESCRIPTION": "My_Quant_Project Trading Market Data canonical one-shot automatic updater.",
    }


def _same_windows_path(left: object, right: object) -> bool:
    return ntpath.normcase(ntpath.normpath(str(left or ""))) == ntpath.normcase(ntpath.normpath(str(right or "")))


def _status_from_payload(payload: Mapping[str, object], spec: Mapping[str, object]) -> dict[str, object]:
    if not bool(payload.get("installed")):
        return {
            "status": SCHEDULER_STATUS_NOT_INSTALLED,
            "supported": True,
            "installed": False,
            "enabled": False,
            "task_name": spec["task_name"],
            "wake_minutes": spec["wake_minutes"],
            "drift_reasons": (),
        }
    enabled = bool(payload.get("enabled"))
    drift: list[str] = []
    if str(payload.get("execute") or "").strip().lower() != str(spec["execute"]).strip().lower():
        drift.append("execute")
    if str(payload.get("arguments") or "").strip() != str(spec["arguments"]).strip():
        drift.append("arguments")
    if not _same_windows_path(payload.get("working_directory"), spec["working_directory"]):
        drift.append("working_directory")
    try:
        actual_interval = int(payload.get("interval_minutes"))
    except (TypeError, ValueError):
        actual_interval = None
    if actual_interval != int(spec["wake_minutes"]):
        drift.append("interval_minutes")
    if not bool(payload.get("logon_trigger")):
        drift.append("logon_trigger")
    if drift:
        status = SCHEDULER_STATUS_DRIFTED_ENABLED if enabled else SCHEDULER_STATUS_DRIFTED_DISABLED
    else:
        status = SCHEDULER_STATUS_INSTALLED_ENABLED if enabled else SCHEDULER_STATUS_INSTALLED_DISABLED
    return {
        "status": status,
        "supported": True,
        "installed": True,
        "enabled": enabled,
        "task_name": spec["task_name"],
        "wake_minutes": actual_interval,
        "expected_wake_minutes": int(spec["wake_minutes"]),
        "state": payload.get("state"),
        "next_run_at": payload.get("next_run_at"),
        "last_run_at": payload.get("last_run_at"),
        "last_task_result": payload.get("last_task_result"),
        "missed_runs": payload.get("missed_runs"),
        "logon_trigger": bool(payload.get("logon_trigger")),
        "drift_reasons": tuple(drift),
        "app_path": spec["app_path"],
    }


def get_market_data_scheduler_status(
    project_root: str | Path,
    *,
    python_executable: str | None = None,
    runner: PowerShellRunner | None = None,
    platform_name: str | None = None,
) -> dict[str, object]:
    spec = build_market_data_scheduler_spec(project_root, python_executable=python_executable)
    if (platform_name or os.name) != "nt":
        return {
            "status": SCHEDULER_STATUS_UNSUPPORTED,
            "supported": False,
            "installed": False,
            "enabled": False,
            "task_name": spec["task_name"],
            "wake_minutes": spec["wake_minutes"],
            "drift_reasons": (),
        }
    code, stdout, stderr = _invoke(_STATUS_SCRIPT, {"MQP_TASK_NAME": str(spec["task_name"])}, runner)
    if code != 0:
        return {
            "status": SCHEDULER_STATUS_ERROR,
            "supported": True,
            "installed": False,
            "enabled": False,
            "task_name": spec["task_name"],
            "wake_minutes": spec["wake_minutes"],
            "error": str(stderr or stdout or f"PowerShell exit {code}").strip(),
            "drift_reasons": (),
        }
    try:
        payload = json.loads(str(stdout or "").strip() or "{}")
    except json.JSONDecodeError as exc:
        return {
            "status": SCHEDULER_STATUS_ERROR,
            "supported": True,
            "installed": False,
            "enabled": False,
            "task_name": spec["task_name"],
            "wake_minutes": spec["wake_minutes"],
            "error": f"Task Scheduler status JSON invalid: {exc}",
            "drift_reasons": (),
        }
    return _status_from_payload(payload, spec)


def _require_windows(platform_name: str | None) -> None:
    if (platform_name or os.name) != "nt":
        raise RuntimeError("Market Data automatic scheduler 只支援 Windows Task Scheduler")


def _run_mutation(
    *,
    script: str,
    spec: Mapping[str, object],
    runner: PowerShellRunner | None,
) -> None:
    code, stdout, stderr = _invoke(script, _scheduler_env(spec), runner)
    if code != 0:
        raise RuntimeError(str(stderr or stdout or f"PowerShell exit {code}").strip())


def install_or_update_market_data_scheduler(
    project_root: str | Path,
    *,
    python_executable: str | None = None,
    runner: PowerShellRunner | None = None,
    platform_name: str | None = None,
) -> dict[str, object]:
    _require_windows(platform_name)
    spec = build_market_data_scheduler_spec(project_root, python_executable=python_executable)
    _run_mutation(script=_INSTALL_SCRIPT, spec=spec, runner=runner)
    return get_market_data_scheduler_status(
        project_root,
        python_executable=python_executable,
        runner=runner,
        platform_name="nt",
    )


def set_market_data_scheduler_enabled(
    project_root: str | Path,
    *,
    enabled: bool,
    python_executable: str | None = None,
    runner: PowerShellRunner | None = None,
    platform_name: str | None = None,
) -> dict[str, object]:
    _require_windows(platform_name)
    spec = build_market_data_scheduler_spec(project_root, python_executable=python_executable)
    current = get_market_data_scheduler_status(
        project_root,
        python_executable=python_executable,
        runner=runner,
        platform_name="nt",
    )
    if not bool(current.get("installed")):
        raise RuntimeError("Windows Task Scheduler 尚未安裝 Market Data Auto Sync")
    _run_mutation(script=_ENABLE_SCRIPT if enabled else _DISABLE_SCRIPT, spec=spec, runner=runner)
    return get_market_data_scheduler_status(
        project_root,
        python_executable=python_executable,
        runner=runner,
        platform_name="nt",
    )


def remove_market_data_scheduler(
    project_root: str | Path,
    *,
    python_executable: str | None = None,
    runner: PowerShellRunner | None = None,
    platform_name: str | None = None,
) -> dict[str, object]:
    _require_windows(platform_name)
    spec = build_market_data_scheduler_spec(project_root, python_executable=python_executable)
    _run_mutation(script=_REMOVE_SCRIPT, spec=spec, runner=runner)
    return get_market_data_scheduler_status(
        project_root,
        python_executable=python_executable,
        runner=runner,
        platform_name="nt",
    )


__all__ = [
    "SCHEDULER_STATUS_UNSUPPORTED",
    "SCHEDULER_STATUS_NOT_INSTALLED",
    "SCHEDULER_STATUS_INSTALLED_ENABLED",
    "SCHEDULER_STATUS_INSTALLED_DISABLED",
    "SCHEDULER_STATUS_DRIFTED_ENABLED",
    "SCHEDULER_STATUS_DRIFTED_DISABLED",
    "SCHEDULER_STATUS_ERROR",
    "build_market_data_scheduler_spec",
    "get_market_data_scheduler_status",
    "install_or_update_market_data_scheduler",
    "set_market_data_scheduler_enabled",
    "remove_market_data_scheduler",
]
