"""Run experiment commands with pinned child settings and safe interruption."""

import os
import pathlib
import shlex
import signal
import subprocess
import time


class CommandError(RuntimeError):
    """Report a child command that returned a nonzero exit code."""


def child_environment(root):
    """Return child settings with fixed threads, hashing, buffering, and imports.

    The function copies the parent environment, prepends the repository ``src``
    directory to ``PYTHONPATH``, and does not mutate the parent process.
    """
    environment = os.environ.copy()
    inherited = environment.get("PYTHONPATH")
    source = str(pathlib.Path(root) / "src")
    environment.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONHASHSEED="0",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=source + (os.pathsep + inherited if inherited else ""),
    )
    return environment


def _group_exists(process_group):
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _poll_group(process, process_group, timeout):
    deadline = time.monotonic() + timeout
    while _group_exists(process_group) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.05)
    return _group_exists(process_group)


def _stop_process_group(process):
    """Terminate a child process group, escalating after a bounded wait."""
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if _poll_group(process, process_group, 5.0):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _poll_group(process, process_group, 5.0)
    process.wait()


def run_command(args, root, *, log=print):
    """Run an argument list from the repository root and stream merged output.

    The child starts in its own process group with ``child_environment``.
    Returned lines omit newline characters. Nonzero exit raises ``CommandError``;
    interruption terminates the complete child group before propagating.
    """
    command = [str(argument) for argument in args]
    process = subprocess.Popen(
        command,
        shell=False,
        cwd=root,
        env=child_environment(root),
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    try:
        assert process.stdout is not None
        for output in process.stdout:
            line = output.rstrip("\r\n")
            lines.append(line)
            log(line)
        return_code = process.wait()
    except BaseException:
        _stop_process_group(process)
        raise
    if return_code:
        rendered = shlex.join(command)
        raise CommandError(f"command exited with code {return_code}: {rendered}")
    return lines
