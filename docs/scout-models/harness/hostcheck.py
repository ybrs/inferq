#!/usr/bin/env python3
"""Host-state checks. Guards FAILURE-MODES items 1-7, 42.

Everything here answers one question: is this box in the same state it was in
for the other measurements? A benchmark that silently spans a governor change,
a thermal droop, or a busy HT sibling produces a ranking of scheduling luck.
"""
import json, os, re, subprocess, time

CPUFREQ = "/sys/devices/system/cpu/cpu{}/cpufreq/{}"


def governors(cpus=range(6)):
    """#1: governor, min/max freq and turbo state per pinned CPU."""
    out = {}
    for c in cpus:
        e = {}
        for f in ("scaling_governor", "scaling_min_freq", "scaling_max_freq",
                  "scaling_cur_freq"):
            try:
                e[f] = open(CPUFREQ.format(c, f)).read().strip()
            except OSError:
                e[f] = None
        out[f"cpu{c}"] = e
    try:
        out["no_turbo"] = open("/sys/devices/system/cpu/intel_pstate/no_turbo").read().strip()
    except OSError:
        out["no_turbo"] = None
    return out


def governor_problems(g):
    """A governor that differs between cores, or turbo disabled mid-sweep, is
    a real confound. A uniform `powersave` is not - intel_pstate still boosts -
    so it is recorded rather than treated as fatal."""
    probs = []
    seen = {v["scaling_governor"] for k, v in g.items() if k.startswith("cpu")}
    if len(seen) > 1:
        probs.append(f"pinned CPUs have different governors: {seen}")
    if g.get("no_turbo") == "1":
        probs.append("turbo is disabled (intel_pstate/no_turbo=1)")
    return probs


def cpuset_effective():
    """#2: which CPUs the container may actually use."""
    for p in ("/sys/fs/cgroup/cpuset.cpus.effective", "/sys/fs/cgroup/cpuset/cpuset.effective_cpus"):
        try:
            return open(p).read().strip()
        except OSError:
            continue
    return None


def parse_cpu_list(spec):
    out = set()
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def per_cpu_busy(window=2.0):
    """#3: busy fraction for every logical CPU, so load on HT siblings 6-11 is
    visible. Whole-machine averages hide a sibling pegged at 100%."""
    def snap():
        d = {}
        for line in open("/proc/stat"):
            if re.match(r"^cpu\d+ ", line):
                p = line.split()
                v = [int(x) for x in p[1:]]
                d[int(p[0][3:])] = (sum(v), v[3] + (v[4] if len(v) > 4 else 0))
        return d
    a = snap()
    time.sleep(window)
    b = snap()
    out = {}
    for c in a:
        dt = b[c][0] - a[c][0]
        out[c] = round(1.0 - (b[c][1] - a[c][1]) / dt, 3) if dt else None
    return out


def swap_counters():
    """#5: pswpin/pswpout, to prove no swapping happened during a run."""
    d = {}
    for line in open("/proc/vmstat"):
        k, v = line.split()
        if k in ("pswpin", "pswpout"):
            d[k] = int(v)
    return d


def disk_busy(window=1.0):
    """#6: fraction of the window the busiest disk spent doing IO."""
    def snap():
        d = {}
        for line in open("/proc/diskstats"):
            f = line.split()
            if len(f) >= 13 and not f[2].startswith(("loop", "ram")):
                d[f[2]] = int(f[12])       # ms spent doing IO
        return d
    a = snap()
    time.sleep(window)
    b = snap()
    return {k: round((b[k] - a[k]) / (window * 1000), 3)
            for k in a if k in b and b[k] - a[k] > 0}


def cpu_mhz(cpus=range(6)):
    """#4: mean MHz over the pinned cores, for drift tracking."""
    vals, cur = [], None
    for line in open("/proc/cpuinfo"):
        if line.startswith("processor"):
            cur = int(line.split(":")[1])
        elif line.startswith("cpu MHz") and cur in set(cpus):
            vals.append(float(line.split(":")[1]))
    return round(sum(vals) / len(vals), 1) if vals else None


def coretemp():
    for base in ("/sys/class/hwmon",):
        try:
            for h in os.listdir(base):
                try:
                    name = open(os.path.join(base, h, "name")).read().strip()
                except OSError:
                    continue
                if name in ("coretemp", "k10temp"):
                    for f in sorted(os.listdir(os.path.join(base, h))):
                        if re.match(r"temp\d+_input", f):
                            return int(open(os.path.join(base, h, f)).read().strip()) / 1000
        except OSError:
            pass
    return None


def snapshot(cpus="0-5"):
    """Everything worth stamping into a run record."""
    pinned = sorted(parse_cpu_list(cpus))
    busy = per_cpu_busy()
    return {
        "governors": governors(pinned),
        "cpuset_effective": cpuset_effective(),
        "cpu_busy_pinned": {c: busy.get(c) for c in pinned},
        "cpu_busy_siblings": {c: v for c, v in busy.items() if c not in pinned},
        "cpu_mhz_mean": cpu_mhz(pinned),
        "coretemp_c": coretemp(),
        "swap": swap_counters(),
        "disk_busy": disk_busy(),
        "loadavg": [round(x, 2) for x in os.getloadavg()],
        "wall_time": time.time(),
    }
