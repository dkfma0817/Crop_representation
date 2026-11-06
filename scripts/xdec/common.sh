# shared settings for the xdec queue scripts (sourced)
ROOT=/home/irteam/Crop_representation
R=$ROOT/output/2026-11-05/xdec
QAT_OUT=/home/irteam/low-bit-training-v2/output/2026-11-04/scale-800m/800m-ste-code7.5e-5-compile
QAT_SAVE=$R/qat_pause
QAT_REL=output/2026-11-04/scale-800m/800m-ste-code7.5e-5-compile   # as written in its --output-dir (relative to cwd)
QAT_TMUX=qat_800m
GPU_UUIDS="GPU-95648b96-4be8-f428-8c79-4f652449f277 GPU-91436828-6b46-5c7f-5349-e2d8feff0942"   # physical GPU 2, 3
EXCL=output/2026-11-04/readout_queue/data_check/train_exclude_image_ids.json
ITEMS="smoke_cpu qat_pause smoke_gpu E1 E2 E3 E4 E5 E6 E7 B0_direct R00 recon qat_resume analysis report"
mkdir -p "$R/state" "$R/logs"
ts() { date '+%F %T'; }
event() { printf '%s\t%s\t%s\n' "$(ts)" "$1" "$2" >> "$R/logs/events.tsv"; echo "$(ts) [$1] $2"; }
set_status() {   # set_status <item> <PENDING|RUNNING|DONE|FAILED|SKIPPED> [note]
  local x
  printf '%s\t%s\t%s\n' "$2" "$(ts)" "${3:-}" > "$R/state/$1"
  event "$1" "$2 ${3:-}"
  { printf 'item\tstatus\tupdated\tnote\n'
    for x in $ITEMS; do
      if [ -f "$R/state/$x" ]; then printf '%s\t%s\n' "$x" "$(cat "$R/state/$x")"; else printf '%s\tPENDING\t-\t-\n' "$x"; fi
    done; } > "$R/queue_status.tsv"
}
get_status() { [ -f "$R/state/$1" ] && cut -f1 "$R/state/$1" || echo PENDING; }
qat_pid() {      # torchrun main pid of THIS QAT job (port 30871, its output dir); never matches our own shell
  ps -eo pid,args | grep -v grep | grep "torch.distributed.run --nproc_per_node 2 --master_port 30871 -m src.qat.train" \
    | grep -- "--output-dir $QAT_REL " | awk '{print $1}' | head -1
}
qat_any() {      # any process of this QAT job (torchrun, workers, dataloader children)
  ps -eo pid,args | grep -v grep | grep "src.qat.train" | grep -- "--output-dir $QAT_REL " | grep -v "master_port 30870" | awk '{print $1}' \
    | while read p; do [ "$(readlink /proc/$p/cwd 2>/dev/null)" = /home/irteam/low-bit-training-v2 ] && \
        tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx 'CUDA_VISIBLE_DEVICES=2,3' && echo $p; done
}
gpu23_apps() {   # compute processes on physical GPU 2/3: "pid used_MiB"
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v u="$GPU_UUIDS" 'BEGIN{n=split(u,a," ");for(i=1;i<=n;i++)ok[a[i]]=1} ok[$2]{print $1, $3}'
}
gpu23_mem() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n 3,4p | tr '\n' ' '; }
qat_step() { tail -c 20000 "$QAT_OUT/train.log" | grep -o '[0-9]*/305176' | tail -1 | cut -d/ -f1; }
qat_last_ckpt() { ls -d "$QAT_OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1; }
