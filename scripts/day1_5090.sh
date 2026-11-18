#!/usr/bin/env bash
# 5090 day 1 in stages (each one resumable; native Isaac Sim, no docker). Usage: bash scripts/day1_5090.sh [stage ...]    default: setup isaac hunyuan
#   setup    torch env, tests, smoke train                              (scripts/setup_5090.sh)
#   isaac    find/install native Isaac Sim, render the hello scene      (scripts/setup_isaac_native_5090.sh)  -> writes .isaac_env
#   hunyuan  Hunyuan3D-2mini env + smoke asset                          (scripts/setup_hunyuan_5090.sh)
#   llm      layout-LLM validity test; needs a served model: LLM_SPEC=openai:<model>@http://localhost:8000/v1 bash scripts/day1_5090.sh llm
#   sets     layouts -> Isaac renders -> effects -> trainer layout for indoor/outdoor/race (N layouts per domain, default 30; HOURS, run overnight)
#   assets   catalogue growth on one frame: FRAME=<png> bash scripts/day1_5090.sh assets   (Ollama + qwen3-vl:8b-instruct must be available)
#   train    teacher cache + training on data/train_sim (configs/sim_5090.yaml, student vit then hybrid) after preflight
# One GPU job at a time: Isaac, Ollama/vLLM, Hunyuan and training never together. Run from the project folder.
set -euo pipefail
cd "$(dirname "$0")/.."; PY=.venv/bin/python; N=${N:-30}
stages=("$@"); [ ${#stages[@]} -gt 0 ] || stages=(setup isaac hunyuan)
need_isaac() { [ -f .isaac_env ] || { echo "run the 'isaac' stage first (.isaac_env missing)"; exit 1; }; set -a; . ./.isaac_env; set +a; export ISAAC_NATIVE=1 PFM_ROOT=$PWD; }
gpu_idle() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -lt 2000 ] || { echo "GPU busy ($u MiB used): stop other GPU jobs ($1 needs an idle GPU)"; exit 1; }; }
for s in "${stages[@]}"; do
  echo "=== stage: $s ($(date +%H:%M:%S))"
  case $s in
    setup) bash scripts/setup_5090.sh ;;
    isaac) gpu_idle isaac; bash scripts/setup_isaac_native_5090.sh ;;
    hunyuan) gpu_idle hunyuan; bash scripts/setup_hunyuan_5090.sh ;;
    llm) : "${LLM_SPEC:?set LLM_SPEC, e.g. openai:Qwen/Qwen3-VL-8B-Instruct@http://localhost:8000/v1}"; mkdir -p runs/llm_select; $PY scripts/llm_select_test.py "$LLM_SPEC" "runs/llm_select/$(echo "$LLM_SPEC" | tr -c 'A-Za-z0-9\n' _ | cut -c1-40).json" ;;
    sets) need_isaac; gpu_idle sets; $PY scripts/gen_sim_set.py layouts "$N"
          for d in indoor outdoor race; do
            $PY scripts/gen_sim_set.py render $d; $PY scripts/gen_sim_set.py effects $d; $PY scripts/gen_sim_set.py train $d data/train_sim; done
          echo "sets done: data/train_sim ($(ls data/train_sim/images 2>/dev/null | wc -l) groups)" ;;
    assets) need_isaac; : "${FRAME:?set FRAME=<png>}"; gpu_idle assets; ISAAC_NATIVE=0 $PY scripts/grow_catalogue.py "$FRAME" --known "${KNOWN:-}" --licence "${LICENCE:-unspecified}" ;;
    train) gpu_idle train; $PY scripts/preflight.py configs/sim_5090.yaml 100000; $PY scripts/cache_teachers.py configs/sim_5090.yaml
           for ST in vit hybrid; do sed "s/^student: .*/student: $ST/" configs/sim_5090.yaml > "/tmp/sim_$ST.yaml"; $PY scripts/train.py "/tmp/sim_$ST.yaml"; done ;;
    *) echo "unknown stage $s"; exit 1 ;;
  esac
done
echo "DAY1 STAGES DONE: ${stages[*]}"
