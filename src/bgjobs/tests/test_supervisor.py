"""Per-GPU background supervisor logic with synthetic GPU state and fake jobs:  python src/bgjobs/tests/test_supervisor.py <scratch>"""
import json, os, shutil, subprocess, sys, time
import yaml
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def main(scr):
    root = os.path.join(scr, "bg")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    smi = os.path.join(scr, "smi.json")
    crop = os.path.join(scr, "cropctl"); shutil.rmtree(crop, ignore_errors=True); os.makedirs(crop)
    gp = [dict(index=i, uuid=f"U{i}", mem_used=0, mem_total=143771, util=0) for i in (0, 1)]

    def w(apps=(), mem=None, util=None):
        g = json.loads(json.dumps(gp))
        for x in g:
            x["mem_used"] = (mem or {}).get(x["index"], 0); x["util"] = (util or {}).get(x["index"], 0)
        json.dump(dict(gpus=g, apps=list(apps)), open(smi + ".t", "w")); os.replace(smi + ".t", smi)
    coex = subprocess.Popen([py_exe := sys.executable, "-c", "import time; time.sleep(600)", "src.crop_aware_pretrain.train"], start_new_session=True)
    w([dict(uuid="U1", pid=999990, mem=5000), dict(uuid="U0", pid=coex.pid, mem=9000)], {0: 9000, 1: 5000}, {0: 90, 1: 90})
    base = yaml.safe_load(open(os.path.join(ROOT, "configs/bgjobs/base.yaml")))
    base["eval"]["steps"] = [5000, 10000, 20000]
    base["eval"]["stop_rule"]["enforce"] = True
    b0 = os.path.join(scr, "base0"); os.makedirs(b0, exist_ok=True); json.dump(dict(val_obj=50.0, val_part=50.0), open(f"{b0}/DONE.json", "w"))
    base["eval"]["baseline_step0_official"] = b0
    bp = os.path.join(scr, "base.yaml"); yaml.safe_dump(base, open(bp, "w"))
    q = dict(spill={1: ["C"]}, gpus={0: ["A", "A2"], "0b": ["C"], 1: ["B"], "1b": ["E"]}, jobs=dict(A=dict(set=[]), A2=dict(set=[]), B=dict(set=[]), C=dict(set=[]), E=dict(set=[], after="A", delete_after_checkpoints=True)))
    qp = os.path.join(scr, "q.yaml"); yaml.safe_dump(q, open(qp, "w"))
    py = sys.executable
    test = dict(root=root, queue=qp, base=bp, crop_ctl=crop, trust=False, sc=dict(poll_s=0.2, idle_stable_s=0.5, external_grace_s=1.5, auto_release_idle_s=3, pause_auto_release_idle_s=3, stop_timeout_s=10, hang_timeout_s=60, max_failures=3, failure_backoff_s=0.3, progress_reset_steps=10 ** 9),
                train_cmd=[py, os.path.join(HERE, "fake_train.py")], eval_cmd=[py, os.path.join(HERE, "fake_eval.py")])
    tj = os.path.join(scr, "t.json"); json.dump(test, open(tj, "w"))
    env = dict(os.environ, CROP_AWARE_FAKE_SMI=smi, PYTHONPATH=ROOT)

    def spawn(n):
        return subprocess.Popen([py, "-u", "-m", "src.bgjobs.supervisor", "--test-json", tj], env=env, cwd=ROOT, stdout=open(f"{scr}/sup{n}.out", "w"), stderr=subprocess.STDOUT)

    def status():
        try:
            return json.load(open(f"{root}/control/status.json"))["slots"]
        except Exception:
            return {}

    def wait(f, t=30, msg=""):
        t0 = time.time()
        while time.time() - t0 < t:
            try:
                if f():
                    return
            except Exception:
                pass
            time.sleep(0.1)
        raise AssertionError("timeout: " + msg)

    def train(g):
        return (status().get(str(g)) or {}).get("train")

    def st(g, job):
        return next(j["status"] for j in status()[str(g)]["jobs"] if j["job"] == job)

    def launches(j):
        p = f"{root}/{j}/launches.txt"
        return open(p).read().splitlines() if os.path.exists(p) else []

    ok = []

    def check(s):
        ok.append(s); print("PASS", s, flush=True)

    ctl_set = {}

    def ctl(*a):
        pass
    for j in "A A2 B C E".split():
        for s in ("control", "logs", "evals", "checkpoints", "data"):
            os.makedirs(f"{root}/{j}/{s}", exist_ok=True)

    sup = spawn(1)
    wait(lambda: train(0) == "A", msg="GPU 0 idle, A starts")
    wait(lambda: launches("A"), msg="A launch recorded")
    assert train(1) is None and "gpu=0" in launches("A")[0]
    check("crop-aware processes on GPU 0 do not block A; an unrelated process on GPU 1 still blocks B")
    wait(lambda: train("0b") == "C" and launches("C"), msg="second slot on GPU 0")
    assert "gpu=0" in launches("C")[0] and train(0) == "A"
    check("slot 0b runs C on GPU 0 alongside A")
    os.makedirs(f"{root}/A/checkpoints/step_00000001", exist_ok=True); os.makedirs(f"{root}/A/evals/step_00005000", exist_ok=True); open(f"{root}/A/evals/step_00005000/keep", "w").write("x")
    assert train("1b") is None
    check("a job with `after` waits for that job's training")
    coex.kill(); coex.wait()
    w([dict(uuid="U1", pid=999990, mem=5000)], {1: 5000}, {1: 90})
    p = subprocess.run([py, "-u", "-m", "src.bgjobs.supervisor", "--test-json", tj], env=env, cwd=ROOT, capture_output=True, timeout=30)
    assert p.returncode == 75, p.returncode
    assert open(f"{root}/control/supervisor.lock").read() == str(sup.pid)
    check("second supervisor refused (lock)")

    w()
    wait(lambda: train(1) == "B", msg="GPU 1 idle, B starts")
    wait(lambda: launches("B"), msg="B launch recorded")
    assert train(0) == "A" and "gpu=1" in launches("B")[0]
    check("both GPUs idle -> A on GPU 0 and B on GPU 1, independent")

    def pause(t, on):
        f = f"{root}/control/PAUSE_{t}"
        open(f, "w").write("x") if on else os.remove(f)
    pause("gpu0", True)
    wait(lambda: train(0) is None and st(0, "A") == "PAUSED", msg="pause gpu0")
    n = len(launches("A")); time.sleep(1.0)
    assert train(1) == "B" and len(launches("A")) == n
    check("pause gpu0 -> A checkpoints and stops during idle grace; B unaffected")
    pause("gpu0", False)
    wait(lambda: train(0) == "A", msg="resume gpu0")
    check("resume gpu0 -> A restarts")

    pause("gpu0", True)
    wait(lambda: train(0) is None, msg="per-GPU pause stops A")
    wait(lambda: train(0) == "A" and not status()["0"]["paused"], t=15, msg="idle per-GPU pause expires")
    assert os.path.exists(f"{root}/control/PAUSE_gpu0")
    check("stale per-GPU pause -> idle GPU resumes automatically")
    pause("gpu0", True)
    wait(lambda: train(0) is None, msg="rewritten pause is honoured")
    pause("gpu0", False)
    wait(lambda: train(0) == "A", msg="rewritten pause removed")
    check("a new pause request takes effect after an earlier pause expired")

    w([dict(uuid="U1", pid=999991, mem=5000)], {1: 5000})
    wait(lambda: train(1) is None, msg="foreground occupies GPU1")
    pause("all", True)
    wait(lambda: train(0) is None, msg="global pause stops A")
    wait(lambda: train(0) == "A" and not status()["0"]["paused"], t=15, msg="global pause expires on idle GPU0")
    assert status()["1"]["paused"] and train(1) is None
    check("global pause -> idle GPU0 resumes while occupied GPU1 stays paused")
    w()
    wait(lambda: train(1) == "B" and not status()["1"]["paused"], t=15, msg="GPU1 resumes after foreground exits")
    check("global pause -> GPU1 resumes automatically when its foreground finishes")

    before = json.load(open(f"{root}/control/active.json"))
    sup.kill(); sup.wait()
    sup = spawn(2)
    wait(lambda: train(0) == "A" and train(1) == "B", msg="adopt jobs after supervisor restart")
    wait(lambda: os.path.getsize(f"{scr}/sup2.out") > 0, msg="restarted supervisor started")
    assert "adopted" in open(f"{scr}/sup2.out").read()
    after = json.load(open(f"{root}/control/active.json"))
    assert all(before[str(g)]["train"]["pids"] == after[str(g)]["train"]["pids"] for g in (0, 1))
    assert not status()["0"]["paused"] and not status()["1"]["paused"]
    pause("all", False)
    check("supervisor restart preserves expired pauses and adopts existing jobs")

    pid0 = json.load(open(f"{root}/control/active.json"))["1"]["train"]["pids"]
    w([dict(uuid="U1", pid=999991, mem=5000)], {1: 5000})
    time.sleep(0.8); w(); time.sleep(1.5)
    assert train(1) == "B" and json.load(open(f"{root}/control/active.json"))["1"]["train"]["pids"] == pid0
    check("external job shorter than the grace period -> B keeps running")
    w([dict(uuid="U1", pid=999991, mem=5000)], {1: 5000})
    wait(lambda: train(1) is None, msg="long external job -> B yields")
    n = len(launches("B")); time.sleep(1.0); assert len(launches("B")) == n and train(0) == "A"
    check("external job stays -> B checkpoints and yields, no repeated launches, A unaffected")
    w()
    wait(lambda: train(1) == "B", msg="B resumes when idle")
    check("GPU 1 idle again -> B resumes")

    json.dump(dict(gpu=1, until=None, note="q"), open(f"{crop}/reserve_gpu_1.json", "w"))
    wait(lambda: train(1) is None, msg="reservation in the crop-aware control dir")
    check("existing queues' reservation (crop-aware control dir) makes B stop")
    if os.path.exists(f"{crop}/reserve_gpu_1.json"):
        os.remove(f"{crop}/reserve_gpu_1.json")
    wait(lambda: train(1) == "B", msg="released")
    check("reservation released -> B resumes")

    json.dump(dict(gpu=1, until=None, note="left behind"), open(f"{crop}/reserve_gpu_1.json", "w"))
    wait(lambda: train(1) is None, msg="stale reservation stops B")
    wait(lambda: train(1) == "B" and not os.path.exists(f"{crop}/reserve_gpu_1.json"), t=15, msg="auto release")
    check("reservation left on an idle GPU -> auto-released, B resumes")
    json.dump(dict(gpu=1, until=time.time() + 3600, note="timed hold"), open(f"{crop}/reserve_gpu_1.json", "w"))
    wait(lambda: train(1) is None, msg="timed reservation stops B")
    time.sleep(5)
    assert train(1) is None and os.path.exists(f"{crop}/reserve_gpu_1.json"), "a timed reservation must be honoured"
    os.remove(f"{crop}/reserve_gpu_1.json")
    wait(lambda: train(1) == "B", msg="timed reservation removed")
    check("reservation with a time limit is honoured on an idle GPU")

    open(f"{root}/B/crash", "w").write("x"); os.kill(pid0[0] if False else json.load(open(f"{root}/control/active.json"))["1"]["train"]["pids"][0], 15)
    wait(lambda: os.path.exists(f"{root}/B/FAILED.json"), t=40, msg="B fails after limited retries")
    n = len(launches("B")); time.sleep(1.2); assert len(launches("B")) == n and st(1, "B") == "FAILED"
    check("repeated crash -> FAILED after limited retries, not restarted, cause recorded")

    cq = yaml.safe_load(open(qp)); cq["jobs"]["A"]["set"] = ["test.done_at=999999999"]; yaml.safe_dump(cq, open(qp, "w"))
    open(f"{root}/A/fake_step", "w").write("0")
    os.remove(f"{root}/B/FAILED.json") ; os.remove(f"{root}/B/crash"); os.remove(f"{root}/B/control/failures.json")
    # gate at 20k: fake evals return no gain -> STOPPED_NO_GAIN
    for s, v in ((5000, 50.1), (10000, 50.0), (20000, 50.1)):
        d = f"{root}/B/evals/step_{s:08d}/official"; os.makedirs(d, exist_ok=True); json.dump(dict(val_obj=v, val_part=v), open(f"{d}/DONE.json", "w"))
    wait(lambda: train(1) == "B", msg="B back")
    d20 = f"{root}/B/evals/step_00020000"; open(f"{d20}/PENDING", "w").write("x")
    wait(lambda: os.path.exists(f"{root}/B/STOPPED_NO_GAIN.json"), t=25, msg="no gain at the gate -> STOPPED_NO_GAIN")
    wait(lambda: train(1) is None and st(1, "B") == "STOPPED_NO_GAIN", t=15, msg="B stops")
    n = len(launches("B")); time.sleep(1.2); assert len(launches("B")) == n
    check("official gain below the rule at the gate -> STOPPED_NO_GAIN, stopped, not restarted")
    wait(lambda: train(1) == "C" and train("0b") is None, t=25, msg="C moves to GPU 1")
    wait(lambda: any("gpu=1" in l for l in launches("C")), msg="C launch on GPU 1")
    check("GPU 1 runs out of its own jobs -> queued control moves there instead of idling")

    cq["jobs"]["A"]["set"] = ["test.done_at=30"]; yaml.safe_dump(cq, open(qp, "w"))
    os.remove(f"{root}/A/control/last_train_exit.json") if False else None
    pid = json.load(open(f"{root}/control/active.json"))["0"]["train"]["pids"][0]
    os.kill(pid, 15); wait(lambda: train(0) in ("A", "A2") and json.load(open(f"{root}/control/active.json"))["0"]["train"]["pids"][0] != pid, msg="A relaunch with done_at")
    wait(lambda: os.path.exists(f"{root}/A/done.json"), t=20, msg="A completes")
    wait(lambda: train(0) == "A2", t=15, msg="A2 starts on GPU 0")
    wait(lambda: launches("A2"), msg="A2 launch recorded")
    assert st(0, "A") == "COMPLETED" and "gpu=0" in launches("A2")[0]
    wait(lambda: train("1b") == "E", t=15, msg="E starts after A")
    assert not os.path.exists(f"{root}/A/checkpoints/step_00000001") and os.path.exists(f"{root}/A/evals/step_00005000/keep")
    check("after A finishes -> E starts, A's checkpoints deleted, A's evals kept")
    n = len(launches("A")); time.sleep(1); assert len(launches("A")) == n
    check("A COMPLETED -> not restarted; next job in GPU 0's list starts")

    json.dump(dict(gpu=0, until=None, note="q"), open(f"{root}/control/reserve_gpu_0.json", "w"))
    wait(lambda: train(0) is None, msg="own reservation")
    os.remove(f"{root}/control/reserve_gpu_0.json")
    wait(lambda: train(0) == "A2", msg="release own")
    cq["gpus"][0] = ["A"]; cq["gpus"]["0b"] = []; cq["gpus"]["1b"] = []; cq["spill"] = {}; yaml.safe_dump(cq, open(qp, "w"))
    wait(lambda: train(0) is None, t=15, msg="job removed from the list stops")
    time.sleep(1); assert all(train(g) is None for g in (0, 1))
    check("no remaining job on a GPU -> stays idle, nothing is invented")
    sup.terminate(); sup.wait()
    print(f"\nALL {len(ok)} CHECKS PASSED")


if __name__ == "__main__":
    main(sys.argv[1])
