#!/usr/bin/env python3
"""Process control and pre-flight gates for the eval harness.

Three rules this module exists to enforce, because breaking any of them
produces numbers that look fine and are wrong:

  1. Every child is tracked by the real PID from Popen and killed through its
     own process group. Nothing here matches process names. `pkill -f llama`
     matches the shell that is running the harness, and `pkill -x llama-server`
     cannot tell our server from somebody else's.
  2. No single step may exceed STEP_TIMEOUT. A watchdog thread enforces it even
     when the main thread is blocked in a socket read, because a socket timeout
     alone does not cover a server wedged mid-generation.
  3. Weights are loaded with `--load-mode none` - no mmap, read straight into
     process memory - and a warmup request is discarded before any timing is
     recorded. Under mmap the first forward pass measures page faults from disk
     rather than compute. Note `--no-mmap` is deprecated in this build and
     `-lm none` is its replacement; passing the old flag logs a deprecation
     warning, which is exactly the kind of silently-ignored setting this
     harness has to rule out, so resident_bytes() checks it actually happened.
"""
import json, os, signal, socket, subprocess, sys, threading, time, urllib.error, urllib.request

STEP_TIMEOUT = int(os.environ.get("EVAL_STEP_TIMEOUT", "300"))
LIVENESS_POLL = 15
MAX_CPU_BUSY = float(os.environ.get("EVAL_MAX_CPU_BUSY", "0.20"))
MIN_FREE_GB = float(os.environ.get("EVAL_MIN_FREE_GB", "8"))


class FatalRunError(RuntimeError):
    """This request or model failed. Recorded, never scored as a zero."""


class StepTimeout(FatalRunError):
    """A step exceeded its wall-clock cap.

    Raised rather than aborting the process: a model whose prefill genuinely
    needs longer than the cap is a finding about that model - too slow for the
    role - not a reason to throw away the other twenty-five runs. The server is
    killed first, which is what unblocks the socket the caller is sitting on.
    Only a global deadline, or a failure to kill, aborts everything.
    """


# --------------------------------------------------------------------------
# child process registry - every PID we create, so abort can clean up
# --------------------------------------------------------------------------
_CHILDREN = {}          # pid -> Popen
_CHILDREN_LOCK = threading.Lock()


def _register(proc):
    with _CHILDREN_LOCK:
        _CHILDREN[proc.pid] = proc


def _unregister(pid):
    with _CHILDREN_LOCK:
        _CHILDREN.pop(pid, None)


def kill_pgid(proc, grace=20):
    """Kill a child and everything it spawned, by process-group id.

    start_new_session=True gave the child its own group, so one killpg reaches
    any thread pool or helper it forked. Returns True if it is gone.
    """
    if proc is None or proc.poll() is not None:
        _unregister(getattr(proc, "pid", None))
        return True
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        _unregister(proc.pid)
        return True
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        deadline = time.time() + (grace if sig == signal.SIGTERM else 10)
        while time.time() < deadline:
            if proc.poll() is not None:
                _unregister(proc.pid)
                return True
            time.sleep(0.2)
    _unregister(proc.pid)
    return proc.poll() is not None


def kill_all_children():
    with _CHILDREN_LOCK:
        procs = list(_CHILDREN.values())
    for p in procs:
        kill_pgid(p, grace=5)


def abort(msg):
    """Kill everything we started and leave. Never return."""
    sys.stderr.write(f"\n!!! ABORT: {msg}\n")
    sys.stderr.flush()
    kill_all_children()
    os._exit(1)


# --------------------------------------------------------------------------
# hard step timeout
# --------------------------------------------------------------------------
class step:
    """Context manager enforcing a wall-clock cap on one step of the run.

    The watchdog is a timer thread rather than a signal alarm so it still fires
    while the main thread is blocked inside urllib, and it calls abort() which
    uses os._exit - a wedged llama-server must not get the chance to be waited
    on by interpreter shutdown.
    """

    def __init__(self, name, timeout=STEP_TIMEOUT):
        self.name = name
        self.timeout = timeout
        self.t0 = None
        self._timer = None

    def __enter__(self):
        self.t0 = time.monotonic()          # 7: monotonic, never wall clock
        self.fired = False
        self._timer = threading.Timer(self.timeout, self._fire)
        self._timer.daemon = True
        self._timer.start()
        return self

    def _fire(self):
        """Kill the children so the blocked socket read returns, then mark it."""
        self.fired = True
        kill_all_children()

    def __exit__(self, et, ev, tb):
        self._timer.cancel()
        self.elapsed = time.monotonic() - self.t0
        if self.fired:
            # Whatever the body raised was a consequence of the kill; report the
            # cause rather than the symptom.
            raise StepTimeout(f"step {self.name!r} exceeded {self.timeout}s "
                              f"(server killed)") from None
        return False


# --------------------------------------------------------------------------
# pre-flight: refuse to measure anything on a box that is not idle
# --------------------------------------------------------------------------
def find_processes_by_exe(exe_path):
    """PIDs whose executable really is exe_path, resolved through /proc/<pid>/exe.

    This compares inodes, not command-line text, so it cannot be fooled by a
    shell whose arguments happen to contain the word 'llama'.
    """
    try:
        target = os.path.realpath(exe_path)
    except OSError:
        return []
    found = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if os.path.realpath(f"/proc/{pid}/exe") == target:
                found.append(int(pid))
        except (OSError, PermissionError):
            continue
    return found


def port_is_open(port, host="127.0.0.1"):
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        s.close()


def cpu_quota():
    """Docker cgroup CPU limit in whole cores, or None if unlimited.

    A container capped below the core count we pin to would make every rate
    look low for a reason that has nothing to do with the model.
    """
    for path, div in (("/sys/fs/cgroup/cpu.max", None),
                      ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us")):
        try:
            raw = open(path).read().strip()
        except OSError:
            continue
        if div is None:
            q, p = raw.split()
            if q == "max":
                return None
            return int(q) / int(p)
        q = int(raw)
        if q <= 0:
            return None
        return q / int(open(div).read().strip())
    return None


def cpu_busy_fraction(window=2.0):
    """Fraction of CPU time that is not idle, sampled over `window` seconds.

    Load average is the obvious check and the wrong one: it is a ~1-minute
    exponential average, so for minutes after a 6-core job is killed it still
    reads as busy and would block a run on a box that is in fact free. This
    reads /proc/stat twice and reports what the CPUs are doing right now.
    """
    def snap():
        for line in open("/proc/stat"):
            if line.startswith("cpu "):
                v = [int(x) for x in line.split()[1:]]
                idle = v[3] + (v[4] if len(v) > 4 else 0)   # idle + iowait
                return sum(v), idle
        return None, None
    t0, i0 = snap()
    time.sleep(window)
    t1, i1 = snap()
    dt = t1 - t0
    if not dt:
        return None
    return 1.0 - (i1 - i0) / dt


def mem_available_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 ** 2)
    return None


def preflight(llama_bin, port, cpus, expect_cores, wait_for_idle=240):
    """Every reason not to start, checked before a single token is generated.

    Idleness is waited for rather than failed on, up to wait_for_idle seconds:
    the common case is that a previous run has just been killed and the box
    needs a moment, which is not a reason to abandon a sweep.
    """
    problems, notes = [], {}

    strays = find_processes_by_exe(os.path.join(llama_bin, "llama-server"))
    mine = {p.pid for p in _CHILDREN.values()}
    strays = [p for p in strays if p not in mine]
    if strays:
        problems.append(f"llama-server already running (pids {strays}) - kill it first")

    if port_is_open(port):
        problems.append(f"port {port} is already in use")

    deadline = time.time() + wait_for_idle
    busy = cpu_busy_fraction()
    while busy is not None and busy > MAX_CPU_BUSY and time.time() < deadline:
        sys.stderr.write(f"  waiting for idle: CPUs {busy:.0%} busy "
                         f"(need <{MAX_CPU_BUSY:.0%})\n")
        sys.stderr.flush()
        time.sleep(10)
        busy = cpu_busy_fraction()
    notes["cpu_busy_now"] = round(busy, 3) if busy is not None else None
    la1, la5, _ = os.getloadavg()
    notes["loadavg_1m"] = round(la1, 2)      # recorded for provenance, not gated on
    notes["loadavg_5m"] = round(la5, 2)
    if busy is not None and busy > MAX_CPU_BUSY:
        problems.append(f"CPUs {busy:.0%} busy after waiting {wait_for_idle}s "
                        f"(need <{MAX_CPU_BUSY:.0%}) - something else is on the box")

    avail = mem_available_gb()
    notes["mem_available_gb"] = round(avail, 1) if avail else None
    if avail is not None and avail < MIN_FREE_GB:
        problems.append(f"only {avail:.1f} GB available - need {MIN_FREE_GB} GB for --no-mmap")

    q = cpu_quota()
    notes["cgroup_cpu_quota"] = q
    if q is not None and q < expect_cores:
        problems.append(f"cgroup CPU quota {q} cores < the {expect_cores} cores we pin to")

    # taskset must land on distinct physical cores, not HT siblings, or every
    # rate is roughly halved for a reason unrelated to the model.
    core_of, want = {}, set()
    for spec in cpus.split(","):
        if "-" in spec:
            a, b = spec.split("-")
            want.update(range(int(a), int(b) + 1))
        else:
            want.add(int(spec))
    try:
        out = subprocess.run(["lscpu", "-p=CPU,CORE"], capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            if line.startswith("#"):
                continue
            c, core = line.split(",")[:2]
            core_of[int(c)] = int(core)
        phys = {core_of[c] for c in want if c in core_of}
        notes["pinned_cpus"] = sorted(want)
        notes["distinct_physical_cores"] = len(phys)
        if len(phys) != len(want):
            problems.append(f"taskset {cpus} covers {len(want)} logical CPUs but only "
                            f"{len(phys)} physical cores - HT siblings are being counted twice")
    except Exception as e:
        notes["lscpu_error"] = str(e)

    return problems, notes


# --------------------------------------------------------------------------
# the server
# --------------------------------------------------------------------------
class LlamaServer:
    """One llama-server, owned by PID, with the config the eval depends on.

    The flags are not incidental. -np 1 keeps the KV cache in one slot (the
    default of 4 splits it and runs unified attention, which measured an 8x
    decode slowdown). --no-mmap loads the weights into RAM so timings measure
    compute and not page faults. -fa on turns on flash attention, which
    dominates at the ~7k context depth the 46-tool block creates.
    """

    def __init__(self, llama_bin, model_path, port, cpus, threads, ctx, logpath,
                 extra=(), fa="on", load_mode="none", cache_type=None):
        self.llama_bin, self.model_path, self.port = llama_bin, model_path, port
        self.cpus, self.threads, self.ctx = cpus, threads, ctx
        self.fa, self.load_mode, self.cache_type = fa, load_mode, cache_type
        self.logpath, self.extra = logpath, list(extra)
        self.proc = self.log = None
        self._watch = None
        self._stop_watch = threading.Event()

    @property
    def cmd(self):
        c = ["taskset", "-c", self.cpus, os.path.join(self.llama_bin, "llama-server"),
             "-m", self.model_path,
             "-t", str(self.threads), "-c", str(self.ctx),
             "-np", "1", "-fa", self.fa, "-lm", self.load_mode,
             "--jinja", "--host", "127.0.0.1", "--port", str(self.port)]
        if self.cache_type:
            c += ["--cache-type-k", self.cache_type, "--cache-type-v", self.cache_type]
        return c + self.extra

    def start(self):
        if port_is_open(self.port):
            raise FatalRunError(f"port {self.port} busy before start; refusing to "
                                f"measure a server that may not be ours")
        self.log = open(self.logpath, "w")
        self.proc = subprocess.Popen(self.cmd, stdout=self.log,
                                     stderr=subprocess.STDOUT,
                                     start_new_session=True)   # own process group
        _register(self.proc)
        return self.proc.pid

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def wait_healthy(self, timeout):
        """Poll /health, failing fast if the PID dies rather than waiting it out."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.alive():
                raise FatalRunError(
                    f"llama-server pid {self.proc.pid} exited with "
                    f"{self.proc.returncode} during load; see {self.logpath}")
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if json.loads(r.read().decode()).get("status") == "ok":
                        return True
            except Exception:
                pass
            time.sleep(0.5)
        raise FatalRunError(f"llama-server pid {self.proc.pid} not healthy after {timeout}s")

    def start_liveness_watch(self):
        """Abort the sweep the moment the server dies, instead of collecting
        a directory full of connection errors that later read as zeros."""
        def loop():
            while not self._stop_watch.wait(LIVENESS_POLL):
                if self.proc is not None and self.proc.poll() is not None:
                    abort(f"llama-server pid {self.proc.pid} died "
                          f"(exit {self.proc.returncode}); see {self.logpath}")
        self._watch = threading.Thread(target=loop, daemon=True)
        self._watch.start()

    def get(self, path, timeout=30):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}",
                                        timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            raise FatalRunError(f"{path} -> {type(e).__name__}: {e}") from None

    def post(self, path, payload, timeout=120):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                     json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise FatalRunError(f"{path} -> HTTP {e.code}: {e.read().decode()[:400]}") from None
        except Exception as e:
            raise FatalRunError(f"{path} -> {type(e).__name__}: {e}") from None

    def props(self):
        """#9: what the server actually configured, not what we asked for."""
        return self.get("/props")

    def assert_props(self, want_ctx):
        p = self.props()
        got_slots = p.get("total_slots")
        got_ctx = (p.get("default_generation_settings") or {}).get("n_ctx")
        if got_slots != 1:
            raise FatalRunError(f"/props total_slots={got_slots}, expected 1")
        if got_ctx and int(got_ctx) != int(want_ctx):
            raise FatalRunError(f"/props n_ctx={got_ctx}, expected {want_ctx}")
        return p

    def mmapped_files(self):
        """#9/#5: is the gguf file-backed in this process's address space?

        RSS alone cannot answer it - for these models the KV cache dwarfs the
        weights - so residency is checked by looking for the file itself in
        /proc/<pid>/maps. With -lm none it must not appear.
        """
        try:
            return [l.split()[-1] for l in open(f"/proc/{self.proc.pid}/maps")
                    if l.rstrip().endswith(".gguf")]
        except OSError:
            return None

    def assert_not_mmapped(self):
        m = self.mmapped_files()
        if m:
            raise FatalRunError(f"weights are mmapped despite -lm none: {m[:2]}")
        return True

    def chat_format(self):
        """#13/#17: the tool-call format llama.cpp chose for this template.

        Exposed on /props rather than logged. "Generic" here means llama.cpp
        could not use the model's own tool syntax and fell back, which is a
        different capability from native tool calling and must not share a
        score column with it.
        """
        try:
            p = self.props()
        except FatalRunError:
            return None
        for k in ("chat_format", "chat_template_format"):
            if k in p:
                return p[k]
        b = p.get("bos_token")  # touch to keep shape stable across versions
        return (p.get("default_generation_settings") or {}).get("chat_format")

    def resident_bytes(self):
        """RSS of the server process, from /proc/<pid>/statm.

        With -lm none the weights are read into the process, so RSS should
        exceed the gguf size once loaded. Under mmap it starts far smaller and
        grows as pages fault in - which is the state that makes the first timed
        request measure disk instead of compute.
        """
        try:
            pages = int(open(f"/proc/{self.proc.pid}/statm").read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            return None

    def assert_weights_resident(self, model_path, min_fraction=0.9):
        rss = self.resident_bytes()
        size = os.path.getsize(model_path)
        if rss is None:
            raise FatalRunError("could not read RSS to confirm weights are resident")
        if rss < size * min_fraction:
            raise FatalRunError(
                f"RSS {rss/2**30:.2f} GiB is below {min_fraction:.0%} of the "
                f"{size/2**30:.2f} GiB model - weights are not resident, so the "
                f"first timed request would measure page faults")
        return rss, size

    def slots_config(self):
        """What the server actually did with -np / ctx, read back from its log."""
        try:
            for line in open(self.logpath):
                if "n_slots" in line and "n_ctx_slot" in line:
                    return line.strip()
        except OSError:
            pass
        return None

    def chat(self, payload, timeout=STEP_TIMEOUT):
        """One /v1/chat/completions call. Returns (data, wall_seconds).

        Raises rather than returning a sentinel: a request that failed must
        never end up in the results as a score of zero.
        """
        if not self.alive():
            raise FatalRunError("server is not alive")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            json.dumps(payload).encode(), {"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode()), time.time() - t0
        except urllib.error.HTTPError as e:
            raise FatalRunError(f"HTTP {e.code}: {e.read().decode()[:500]}") from None
        except Exception as e:
            # RemoteDisconnected, URLError, socket.timeout, ConnectionReset - all
            # mean "no answer", and all must be recorded rather than crash the sweep.
            raise FatalRunError(f"{type(e).__name__}: {e}") from None

    def assert_serving(self, expected_path):
        """The response carries the gguf path the server actually loaded.

        Checked on every request, because the failure this catches - a stale
        server answering for the previous model under this model's name - is
        invisible in the results.
        """
        d, _ = self.chat({"messages": [{"role": "user", "content": "hi"}],
                          "max_tokens": 1, "temperature": 0, "stream": False}, timeout=60)
        got = d.get("model", "")
        if os.path.realpath(got) != os.path.realpath(expected_path):
            raise FatalRunError(f"server is serving {got!r}, expected {expected_path!r}")
        return got

    def stop(self):
        self._stop_watch.set()
        ok = kill_pgid(self.proc)
        if self.log:
            self.log.close()
        for _ in range(60):
            if not port_is_open(self.port):
                break
            time.sleep(0.5)
        else:
            raise FatalRunError(f"port {self.port} still open after killing "
                                f"pid {getattr(self.proc, 'pid', '?')}")
        return ok
