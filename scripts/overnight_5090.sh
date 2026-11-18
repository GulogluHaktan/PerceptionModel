#!/usr/bin/env bash
# Unattended chain for the 5090 (no Claude needed): waits for the running LLM test, then (1) evaluates the existing sim_v2 as seed 0, (2) trains vit with seeds 1 and 2 and evaluates them, (3) layout-LLM tests with the new "auto" prompt,
# (4) catalogue growth on 3 real photos, (5) trains hybrid seed 1, then writes ~/overnight_summary.md (mean and sd over seeds). A step is skipped if it cannot finish before DEADLINE (default 22:00). One GPU job at a time.
# Status: cat ~/overnight_status.txt ; results: cat ~/overnight_summary.md ; per-step logs: ~/overnight_<step>.out
set -uo pipefail
cd "$(dirname "$0")/.."; export PATH=$HOME/.local/bin:$PATH I23D=$HOME/Projects/i23d; PY=.venv/bin/python; ST=$HOME/overnight_status.txt; DEADLINE=${DEADLINE:-22:00}; dl=$(date -d "today $DEADLINE" +%s)
log() { echo "$(date +%H:%M:%S) $*" | tee -a $ST; }
free_gpu() { for m in $(ollama ps 2>/dev/null | awk 'NR>1{print $1}'); do ollama stop "$m" 2>/dev/null; done; sleep 8; }
step() { local name=$1 est=$2; shift 2; if [ $(( $(date +%s) + est * 60 )) -gt $dl ]; then log "ATLANDI (sure yetmez): $name"; return 0; fi; log "BASLADI: $name (tahmin ${est} dk)"; free_gpu; "$@" > $HOME/overnight_$name.out 2>&1; log "BITTI(kod $?): $name"; }
eval_all() { local ck=$1 tag=$2 kinds=$3; export EVAL_ROOT=data/heldout CKPT=$ck TAG=$tag KINDS=$kinds
  for pair in "sim_race 4" "sim_racep 4" "sim_outdoor 4" "sim_indoor 1"; do set -- $pair; $PY scripts/eval_students_sim.py $1 $2 2>&1 | grep -E "^(vit|hybrid)|Error|Traceback"; done
  for s in sim_racep sim_racec sim_race; do $PY scripts/eval_slots_sim.py $s 1 2>&1 | grep -E "^(vit|hybrid)|Error|Traceback"; done; }
train_seed() { local seed=$1 kind=$2; local cfg=/tmp/sim_v2_s${seed}_${kind}.yaml; sed -e "s#^out_dir: .*#out_dir: runs/sim_v2s${seed}#" -e "s#^student: .*#student: ${kind}#" configs/sim_v2_5090.yaml > $cfg; echo "seed: ${seed}" >> $cfg; $PY scripts/train.py $cfg; }
log "ZINCIR BASLADI; bitis siniri $DEADLINE"
until grep -a -q LLM5_EXIT $HOME/llm_test5.out 2>/dev/null; do sleep 20; done; log "onceki LLM testi bitti"
step eval_s0 12 bash -c "$(declare -f eval_all); cd $PWD; PY=$PY; eval_all runs/sim_v2 s0 vit,hybrid"
step train_vit_s1 85 bash -c "$(declare -f train_seed); PY=$PY; train_seed 1 vit"
step eval_vit_s1 8 bash -c "$(declare -f eval_all); PY=$PY; eval_all runs/sim_v2s1 s1 vit"
step train_vit_s2 85 bash -c "$(declare -f train_seed); PY=$PY; train_seed 2 vit"
step eval_vit_s2 8 bash -c "$(declare -f eval_all); PY=$PY; eval_all runs/sim_v2s2 s2 vit"
step llm_32b_outdoor 25 env ONLY_DOMAINS=outdoor $PY scripts/llm_select_test.py ollama:qwen3:32b runs/llm_select/qwen3_32b_outdoor_auto2.json
step llm_8b_all 20 $PY scripts/llm_select_test.py ollama:qwen3-vl:8b-instruct runs/llm_select/qwen3vl_8b_full_auto2.json
frames=$(ls data/images/aeroscapes | sed -n '300p;900p;1500p' | sed 's#^#data/images/aeroscapes/#' | tr '\n' ' ')
step grow 40 bash -c "for f in $frames; do $PY scripts/grow_catalogue.py \$f --known '' --max_objects 2 --licence Aeroscapes --out runs/grow_5090 2>&1 | grep -E 'kept|accepted|rejected|duplicate|no new|Error'; done"
step train_hybrid_s1 90 bash -c "$(declare -f train_seed); PY=$PY; train_seed 1 hybrid"
step eval_hybrid_s1 8 bash -c "$(declare -f eval_all); PY=$PY; eval_all runs/sim_v2s1 s1 hybrid"
$PY scripts/summarize_seeds.py > $HOME/overnight_summary.md 2>&1; log "OZET YAZILDI: ~/overnight_summary.md"; log "ZINCIR BITTI"
