import os, signal, sys, time
signal.signal(signal.SIGTERM, lambda *_: sys.exit(9))
d = sys.argv[1]
s = float(os.environ.get("FAKE_EVAL_S", "2"))
open(f"{d}/fake_eval_step_{os.environ['CROP_AWARE_EVAL_STEP']}", "w").write(os.environ["CUDA_VISIBLE_DEVICES"])
time.sleep(s)
