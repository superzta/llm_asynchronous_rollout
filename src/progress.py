"""Lightweight progress printer shared by all runner scripts.

Writes a single-line, flushed-to-stderr progress message per step so that
users watching `tee` output can see the loop is alive. Honors the
``RUNNER_PROGRESS`` env var: 0 = silent, 1 (default) = one line per step,
2 = verbose (additional key/val pairs).
"""
import os
import sys
import time

_PROGRESS_LEVEL = int(os.environ.get("RUNNER_PROGRESS", "1"))
_PROGRESS_EVERY = max(1, int(os.environ.get("RUNNER_PROGRESS_EVERY", "1")))


class ProgressReporter(object):
    def __init__(self, tag, total=None, stream=None):
        self.tag = str(tag)
        self.total = int(total) if total else None
        self.stream = stream if stream is not None else sys.stderr
        self.t0 = time.time()
        self.last_log_t = self.t0
        self.step_count = 0
        self.enabled = _PROGRESS_LEVEL > 0

    def log(self, step=None, **kv):
        if not self.enabled:
            return
        self.step_count += 1
        if (self.step_count % _PROGRESS_EVERY) != 0:
            return
        now = time.time()
        elapsed = now - self.t0
        step_str = "?" if step is None else str(step)
        total_str = "?" if self.total is None else str(self.total)
        parts = ["[%s][%s/%s t=%5.1fs]" % (self.tag, step_str, total_str, elapsed)]
        for k, v in kv.items():
            if isinstance(v, float):
                parts.append("%s=%.3f" % (k, v))
            else:
                parts.append("%s=%s" % (k, v))
        try:
            self.stream.write(" ".join(parts) + "\n")
            self.stream.flush()
        except Exception:
            pass
        self.last_log_t = now

    def note(self, msg):
        if not self.enabled:
            return
        try:
            self.stream.write("[%s] %s\n" % (self.tag, msg))
            self.stream.flush()
        except Exception:
            pass
