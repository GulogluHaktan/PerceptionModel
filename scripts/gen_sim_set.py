"""Batch driver for simulated training data on the 5090 (indoor, outdoor, racing). Layouts come from the rule-based generators (or an LLM later: scripts/llm_select_test.py shows how), every layout is resolved (rules) before it is rendered.
  python scripts/gen_sim_set.py check                 resolve one layout per domain (no GPU)
  python scripts/gen_sim_set.py layouts <N>           write N layouts per domain to runs/{indoor,outdoor,race}_job/layouts/b_<domain>_<i>.json
  python scripts/gen_sim_set.py render <domain>       render every b_<domain>_* layout that has no finished output yet (Isaac, resumable; needs an idle GPU)
  python scripts/gen_sim_set.py effects <domain>      motion blur / fog / rain / low light copies (CPU)
  python scripts/gen_sim_set.py train <domain> <out>  convert to the trainer layout (racing keeps gate-corner files; indoor/outdoor: seg + depth only)
  python scripts/gen_sim_set.py all <N>               layouts + render + effects for the three domains
Domain names: indoor | outdoor | race."""
import glob, json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, "src")
ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT)
JOB = {"indoor": "runs/indoor_job", "outdoor": "runs/outdoor_job", "race": "runs/race_job"}
SH = {"indoor": "sim/run_indoor_set.sh", "outdoor": "sim/run_outdoor_set.sh", "race": "sim/run_race_set.sh"}


def make(domain, i):
    if domain == "indoor":
        from pfm.layout_gen import KINDS, generate
        kinds = list(KINDS); return generate(kinds[i % len(kinds)], 1000 + i)
    if domain == "outdoor":
        from pfm.layout_gen_out import MODES, generate
        return generate(MODES[i % len(MODES)], 1000 + i)
    from pfm.layout_race import generate
    sp = [10, 15, 20, 25, 30][i % 5]; return generate(1000 + i, n_gates=6 + i % 5, speed=sp, setting=["field", "hall"][(i // 5) % 2], light=["noon", "overcast", "dusk"][(i // 2) % 3], offset=[0.0, 0.3, -0.3, 0.5, -0.5][i % 5], focus=8, tilt=[15, 25, 35][i % 3])


def resolve(domain, lay):
    from pfm import layout as a, layout_out as b, layout_race as c
    return {"indoor": a, "outdoor": b, "race": c}[domain].resolve(lay)


def layouts(n):
    for d in JOB:
        Path(JOB[d], "layouts").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            lay = make(d, i); r = resolve(d, lay); assert r["ok"], (d, i, r["errors"][:2]); json.dump(lay, open(f"{JOB[d]}/layouts/b_{d}_{i}.json", "w"))
        print(d, n, "layouts")


def todo(d):
    return [Path(p).stem for p in sorted(glob.glob(f"{JOB[d]}/layouts/b_{d}_*.json")) if not Path(JOB[d], "out", Path(p).stem, "poses.json").exists()]


def render(d):
    names = todo(d); print(d, len(names), "to render"); names and subprocess.run(["bash", SH[d], *names], check=True)


def effects(d):
    runs = [str(Path(p).parent) for p in sorted(glob.glob(f"{JOB[d]}/out/b_{d}_*/poses.json"))]; subprocess.run([sys.executable, "scripts/city_effects.py", f"data/sim_{d}", *runs], check=True)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "check":
        for d in JOB:
            r = resolve(d, make(d, 0)); print(d, "ok" if r["ok"] else r["errors"][:2])
    elif c == "layouts": layouts(int(sys.argv[2]))
    elif c == "render": render(sys.argv[2])
    elif c == "effects": effects(sys.argv[2])
    elif c == "train":
        d, out = sys.argv[2], sys.argv[3]
        subprocess.run([sys.executable, "scripts/race_to_training.py", out, f"data/sim_{d}", f"{JOB[d]}/out", f"sim_{d}"], check=True)   # same converter for all domains: seg + depth; gate files only exist where the index has gates (empty = true negative)
    elif c == "all":
        layouts(int(sys.argv[2]))
        for d in JOB: render(d); effects(d)
