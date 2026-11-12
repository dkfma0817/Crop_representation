"""Supervisor logic test with synthetic GPU state (CROP_AWARE_FAKE_SMI) and a fake trainer: scaling 1<->2 GPUs, preemption, pause, reserve, crash halt, eval, lock, done.

  python src/crop_aware_pretrain/tests/test_supervisor.py <scratch dir>
"""
import json, os, shutil, signal, subprocess, sys, time
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def wait(pred, t=25, msg=""):
    t0 = time.time()
    while time.time() - t0 < t:
        if pred():
            return True
        time.sleep(0.15)
    raise AssertionError(f"timeout waiting for: {msg}")


def main(scr):
    run = os.path.join(scr, "run")
    if os.path.exists(run):
        shutil.rmtree(run)
    for d in ("control", "logs", "evals/step_00000000"):
        os.makedirs(os.path.join(run, d))
    smi = os.path.join(scr, "smi.json")
    gpus = [dict(index=2, uuid="U2", mem_used=0, mem_total=143771, util=0), dict(index=3, uuid="U3", mem_used=0, mem_total=143771, util=0)]
    py = sys.executable
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/crop_aware_pretrain/default.yaml")))
    cfg["run_dir"] = run
    cfg["supervisor"].update(poll_s=0.25, idle_stable_s=0.6, scale_up_stable_s=1.2, min_run_before_rescale_s=1.5, stop_timeout_s=10, rank_grace_s=2, hang_timeout_s=60,
                             failure_backoff_s=0.5, progress_reset_steps=5000, external_grace_s=1.5, watch_gpus=[2, 3], yield_to_external=True,
                             test_train_cmd=[py, os.path.join(HERE, "fake_train.py")],
                             test_eval_cmd=[py, os.path.join(HERE, "fake_eval.py"), run])
    cfg["test_crash"] = False
    cfgp = os.path.join(scr, "cfg.yaml")
    yaml.safe_dump(cfg, open(cfgp, "w"))

    def write(apps=(), mem=None, util=None):
        g = json.loads(json.dumps(gpus))
        for x in g:
            if mem and x["index"] in mem:
                x["mem_used"] = mem[x["index"]]
            if util and x["index"] in util:
                x["util"] = util[x["index"]]
        json.dump(dict(gpus=g, apps=list(apps)), open(smi + ".tmp", "w"))
        os.replace(smi + ".tmp", smi)

    env = dict(os.environ, CROP_AWARE_FAKE_SMI=smi, PYTHONPATH=ROOT)
    write()

    def launches():
        p = os.path.join(run, "launches.txt")
        return open(p).read().splitlines() if os.path.exists(p) else []

    def status():
        p = os.path.join(run, "control/status.json")
        return json.load(open(p)) if os.path.exists(p) else {}

    def ctl(*a):
        subprocess.check_call([py, "-m", "src.crop_aware_pretrain.ctl", *a, "--config", cfgp], env=env, cwd=ROOT, stdout=subprocess.DEVNULL)

    def train_gpus():
        t = status().get("train")
        return (t or {}).get("gpus") if t and not t.get("stopping") else None

    sup = subprocess.Popen([py, "-u", "-m", "src.crop_aware_pretrain.supervisor", "--config", cfgp], env=env, cwd=ROOT,
                           stdout=open(os.path.join(scr, "sup.out"), "w"), stderr=subprocess.STDOUT)
    ok = []

    def check(name):
        ok.append(name)
        print("PASS", name, flush=True)

    wait(lambda: os.path.exists(os.path.join(run, "control/status.json")), msg="supervisor up")
    time.sleep(0.5)
    p2 = subprocess.run([py, "-u", "-m", "src.crop_aware_pretrain.supervisor", "--config", cfgp], env=env, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert p2.returncode == 75, (p2.returncode, p2.stdout[-300:])
    check("second supervisor refuses to start (lock)")

    wait(lambda: train_gpus() == [2, 3], msg="2 idle GPUs -> world 2")
    n0 = len(launches())
    assert any("rank0 gpus=2,3" in l for l in launches()) and any("rank1 gpus=2,3" in l for l in launches()), launches()
    check("two idle GPUs -> DDP world 2 on both (one job, two ranks)")

    pid = status()["train"]["pids"]
    write(apps=[dict(uuid="U2", pid=pid[0], mem=9000), dict(uuid="U3", pid=pid[1], mem=9000)], mem={2: 9000, 3: 9000}, util={2: 90, 3: 90})
    time.sleep(1.5)
    assert train_gpus() == [2, 3] and status()["train"]["pids"] == pid
    check("own processes on the GPUs are not treated as external")

    pid_now = status()["train"]["pids"]
    write(apps=[dict(uuid="U3", pid=999999, mem=5000)], mem={3: 5000})
    time.sleep(0.7)
    write()
    time.sleep(1.5)
    assert train_gpus() == [2, 3] and status()["train"]["pids"] == pid_now, "short external job must not stop training"
    check("external job shorter than the grace period -> training keeps running")
    write(apps=[dict(uuid="U3", pid=999999, mem=5000)], mem={3: 5000})
    wait(lambda: train_gpus() == [2], msg="external process on GPU 3 -> world 1")
    wait(lambda: "gpus=2 " in launches()[-1], msg="world 1 launch recorded")
    check("external process appears -> checkpoint-stop, restart without that GPU (world 2 -> 1)")

    write()
    wait(lambda: train_gpus() == [2, 3], t=20, msg="GPU 3 free again -> world 2")
    check("second GPU stays idle -> checkpoint-stop, restart world 2 (scale up)")

    write(apps=[dict(uuid="U2", pid=888888, mem=5000), dict(uuid="U3", pid=888889, mem=5000)], mem={2: 5000, 3: 5000}, util={2: 90, 3: 90})
    wait(lambda: status().get("train") is None and len(launches()) > 0, msg="both busy -> stop")
    nb = len(launches())
    time.sleep(1.5)
    assert len(launches()) == nb and status().get("train") is None
    check("both GPUs busy -> stopped, no repeated launches while busy")

    write()
    wait(lambda: train_gpus() == [2, 3], msg="idle again -> auto resume")
    check("GPUs idle again -> automatic resume")

    ctl("pause")
    wait(lambda: status().get("train") is None and status().get("paused"), msg="pause")
    nb = len(launches())
    time.sleep(1.5)
    assert len(launches()) == nb
    check("pause -> safe stop, no restart while paused")
    ctl("resume")
    wait(lambda: train_gpus() == [2, 3], msg="resume")
    check("resume -> restarts")

    ctl("reserve", "--gpus", "3")
    wait(lambda: train_gpus() == [2], msg="reserve 3")
    check("reserve GPU 3 -> world 1 without touching other jobs")
    ctl("release", "--gpus", "all")
    wait(lambda: train_gpus() == [2, 3], t=20, msg="release")
    check("release -> world 2")

    d0 = os.path.join(run, "evals/step_00000000")
    open(os.path.join(d0, "PENDING"), "w").write("x")
    wait(lambda: os.path.exists(os.path.join(d0, "DONE")), t=30, msg="eval runs")
    assert open(os.path.join(run, "fake_eval_step_0")).read() in ("2", "3")
    wait(lambda: train_gpus() == [2, 3], t=20, msg="training back on both GPUs")
    check("pending PACO eval -> runs on a GPU alongside training, training keeps world 2")

    ctl("pause")
    wait(lambda: status().get("train") is None, msg="pause2")
    cfg["test_crash"] = True
    yaml.safe_dump(cfg, open(cfgp, "w"))
    open(os.path.join(run, "fake_step"), "w").write("0")
    open(os.path.join(run, "crash_now"), "w").write("1")
    sup.terminate()
    wait(lambda: sup.poll() is not None, msg="supervisor exit")
    check("supervisor stops on SIGTERM")
    sup = subprocess.Popen([py, "-u", "-m", "src.crop_aware_pretrain.supervisor", "--config", cfgp], env=env, cwd=ROOT,
                           stdout=open(os.path.join(scr, "sup2.out"), "w"), stderr=subprocess.STDOUT)
    time.sleep(1.0)
    ctl("resume")
    wait(lambda: os.path.exists(os.path.join(run, "control/HALT_failure.json")), t=40, msg="repeated crash -> halt")
    nb = len(launches())
    time.sleep(1.5)
    assert len(launches()) == nb
    check("repeated crashes -> halted after the failure limit, no more restarts")

    os.remove(os.path.join(run, "crash_now"))
    cfg["test_crash"] = False
    cfg["test_done_at"] = 3
    yaml.safe_dump(cfg, open(cfgp, "w"))
    ctl("stop")
    wait(lambda: sup.poll() is not None, msg="supervisor exit 2")
    ctl("clear-failure")
    os.remove(os.path.join(run, "fake_step"))
    sup = subprocess.Popen([py, "-u", "-m", "src.crop_aware_pretrain.supervisor", "--config", cfgp], env=env, cwd=ROOT,
                           stdout=open(os.path.join(scr, "sup3.out"), "w"), stderr=subprocess.STDOUT)
    wait(lambda: os.path.exists(os.path.join(run, "done.json")), t=30, msg="done")
    n = len(launches())
    time.sleep(2)
    assert len(launches()) == n, "restarted after completion"
    check("training completed -> not restarted")
    sup.send_signal(signal.SIGTERM)
    sup.wait()
    os.remove(os.path.join(run, "done.json"))
    os.remove(os.path.join(run, "fake_step"))
    cfg["test_done_at"] = None
    cfg["supervisor"]["yield_to_external"] = False
    yaml.safe_dump(cfg, open(cfgp, "w"))
    write()
    sup = subprocess.Popen([py, "-u", "-m", "src.crop_aware_pretrain.supervisor", "--config", cfgp], env=env, cwd=ROOT,
                           stdout=open(os.path.join(scr, "sup4.out"), "w"), stderr=subprocess.STDOUT)
    wait(lambda: train_gpus() == [2, 3], msg="training in no-yield mode")
    pid_now = status()["train"]["pids"]
    write(apps=[dict(uuid="U2", pid=777777, mem=5000), dict(uuid="U3", pid=777778, mem=5000)], mem={2: 5000, 3: 5000}, util={2: 90, 3: 90})
    time.sleep(3.5)
    assert train_gpus() == [2, 3] and status()["train"]["pids"] == pid_now, "yield_to_external false must keep training"
    check("yield_to_external false -> external job on both GPUs does not stop training")
    ctl("reserve", "--gpus", "3")
    wait(lambda: train_gpus() == [2] or status().get("train") is None, msg="reserve still honoured")
    check("explicit reserve still stops training on that GPU")
    write()
    wait(lambda: train_gpus() == [2], msg="world 1 on GPU 2 while GPU 3 is reserved")
    write(apps=[dict(uuid="U2", pid=777777, mem=5000)], mem={2: 5000}, util={2: 90})
    ctl("release", "--gpus", "all")
    wait(lambda: train_gpus() == [3], t=20, msg="move to the free GPU")
    check("training on a GPU shared with an external job moves to a free GPU")
    sup.send_signal(signal.SIGTERM)
    sup.wait()
    print(f"\nALL {len(ok)} SUPERVISOR CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
