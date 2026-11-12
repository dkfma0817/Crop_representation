"""GPU state from nvidia-smi: per GPU index, memory, utilization, and which processes hold it. Env CROP_AWARE_FAKE_SMI=<json> substitutes a
synthetic state (supervisor unit tests only): {"gpus": [{"index", "uuid", "mem_used", "mem_total", "util"}], "apps": [{"uuid", "pid", "mem"}]}."""
import json
import os
import subprocess


def _smi(args):
    return subprocess.check_output(["nvidia-smi", f"--query-{args}", "--format=csv,noheader,nounits"], text=True, timeout=120)


def query():
    fake = os.environ.get("CROP_AWARE_FAKE_SMI")
    if fake:
        return json.load(open(fake))
    gpus = []
    for line in _smi("gpu=index,uuid,memory.used,memory.total,utilization.gpu").splitlines():
        i, u, m, t, ut = [x.strip() for x in line.split(",")]
        gpus.append(dict(index=int(i), uuid=u, mem_used=int(m), mem_total=int(t), util=int(ut) if ut.isdigit() else 100))
    apps = []
    for line in _smi("compute-apps=gpu_uuid,pid,used_memory").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 3:
            apps.append(dict(uuid=p[0], pid=int(p[1]), mem=int(p[2]) if p[2].isdigit() else 0))
    return dict(gpus=gpus, apps=apps)


def proc_sid(pid):
    """Session id of a process, or None if it does not exist."""
    try:
        s = open(f"/proc/{pid}/stat").read()
    except OSError:
        return None
    return int(s[s.rindex(")") + 2:].split()[3])


def is_zombie(pid):
    """A dead process that its parent has not reaped: nvidia-smi can keep listing it, but it runs nothing, so it must not count as a GPU user."""
    try:
        s = open(f"/proc/{pid}/stat").read()
    except OSError:
        return False
    return s[s.rindex(")") + 2:].split()[0] == "Z"


def busy_external(index, state, own_sids=()):
    """-> (is_busy, why) for one GPU given the state and the session ids of our own processes. Used by train.py and the supervisor."""
    g = next(x for x in state["gpus"] if x["index"] == index)
    apps = [a for a in state["apps"] if a["uuid"] == g["uuid"]]
    own = [a for a in apps if proc_sid(a["pid"]) in set(own_sids)]
    ext = [a for a in apps if a not in own]
    if ext:
        return True, f"external compute process(es) {[a['pid'] for a in ext]}"
    return False, ""


def describe(index, state, mem_idle, util_idle, own_sids=()):
    """-> dict(busy, why, mem_external). Busy: an external compute process, or external memory above mem_idle, or utilization above util_idle
    while none of our processes is on it (nvidia-smi can miss processes of other containers; memory / utilization catch those)."""
    g = next(x for x in state["gpus"] if x["index"] == index)
    apps = [a for a in state["apps"] if a["uuid"] == g["uuid"]]
    dead = [a for a in apps if is_zombie(a["pid"])]
    own = [a for a in apps if proc_sid(a["pid"]) in set(own_sids) and a not in dead]
    ext = [a for a in apps if a not in own and a not in dead]
    own_mem = sum(a["mem"] for a in own)
    ext_mem = max(0, g["mem_used"] - own_mem - sum(a["mem"] for a in dead))
    why = None
    if ext:
        why = f"external compute process(es) {[a['pid'] for a in ext]}"
    elif ext_mem > mem_idle:
        why = f"memory used by others {ext_mem} MiB > {mem_idle} MiB"
    elif not own and g["util"] > util_idle:
        why = f"utilization {g['util']}% > {util_idle}%"
    return dict(index=index, busy=why is not None, why=why, mem_used=g["mem_used"], mem_external=ext_mem, mem_total=g["mem_total"],
                util=g["util"], own_pids=[a["pid"] for a in own], ext_pids=[a["pid"] for a in ext], mem_own=own_mem)
