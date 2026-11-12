import json, os, sys
a = sys.argv[1:]
run = next(s.split("=", 1)[1] for s in a if s.startswith("run_dir="))
step = int(os.environ["BGJOB_EVAL_STEP"])
d = f"{run}/evals/step_{step:08d}/official"; os.makedirs(d, exist_ok=True)
v = float(os.environ.get("FAKE_AP", "50.0"))
json.dump(dict(step=step, epoch=5, val_obj=v, val_part=v), open(f"{d}/DONE.json", "w"))
