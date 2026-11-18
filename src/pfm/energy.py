"""Orin energy measurement from `tegrastats` (decision 9: accuracy per Orin joule).

Orin NX tegrastats prints rail power like `VDD_IN 5123mW/5100mW` (current/average). We sample the
current value of the total-input rail (VDD_IN) and integrate over wall time. Rail names differ across
JetPack versions: pass `rail` if yours differs. Needs `tegrastats` on PATH (Jetson only; run with sudo
if your JetPack requires it). The parser is unit-tested on sample lines; the live run is NOT yet tested
on an Orin.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time


def parse_power_mw(line: str, rail: str = "VDD_IN"):
    m = re.search(rf"{re.escape(rail)}\s+(\d+)mW(?:/(\d+)mW)?", line)
    return int(m.group(1)) if m else None


class EnergyMeter:
    """with EnergyMeter() as e: run_workload(); e.joules, e.mean_watts, e.seconds"""

    def __init__(self, rail: str = "VDD_IN", interval_ms: int = 100):
        self.rail, self.interval_ms = rail, interval_ms
        self.samples: list[tuple[float, int]] = []
        self._proc = None

    def __enter__(self):
        if shutil.which("tegrastats") is None:
            raise RuntimeError("tegrastats not found: EnergyMeter only works on a Jetson")
        self._proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval_ms)],
                                      stdout=subprocess.PIPE, text=True)
        self._t0 = time.time()

        def reader():
            for line in self._proc.stdout:
                p = parse_power_mw(line, self.rail)
                if p is not None:
                    self.samples.append((time.time(), p))
        self._th = threading.Thread(target=reader, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *a):
        self.seconds = time.time() - self._t0
        self._proc.terminate()
        self._th.join(timeout=1)
        self.joules = self.integrate(self.samples)
        self.mean_watts = self.joules / self.seconds if self.seconds else 0.0

    @staticmethod
    def integrate(samples):
        """Trapezoid rule over (time_s, mW) samples -> joules."""
        j = 0.0
        for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
            j += (p0 + p1) / 2 / 1000 * (t1 - t0)
        return j
