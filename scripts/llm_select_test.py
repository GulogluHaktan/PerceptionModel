"""50-scenario validity test to pick the 5090 LLM (indoor 20, outdoor 15, racing 15). Usage:
  python scripts/llm_select_test.py ollama:qwen3:8b runs/llm_select/qwen3_8b.json [max_repairs]
  python scripts/llm_select_test.py openai:<model>@http://localhost:8000/v1 runs/llm_select/<name>.json
Reports per domain: first-try valid, valid after repairs, mean seconds, and the most common error words. Pick the model with the best valid-after-repair, break ties by speed."""
import collections, json, re, sys, time
from pathlib import Path
sys.path.insert(0, "src")
from pfm.layout_llm_any import make, ollama_ask, openai_ask, DOMAINS
IN = LL = None
REQ = {"indoor": [
    "A small living room, daytime, camera walking slowly from the door side to the sofa.", "A bedroom at night with the lamp on, camera flying low at 1 m height.", "An office with a desk, a chair and a bookcase, camera inverted, moving 2 m/s.",
    "A kitchen with a fridge and a stove, dark lighting.", "A cluttered living room with at least 6 items, one TV standing on a table, daytime.", "A narrow bedroom 3 m by 5 m with a wardrobe and a bed, camera looking at the window.",
    "A large living room 7 m by 6 m with two sofas, a table between them and plants in the corners.", "An office with a sofa in the corner, lamp light, camera fast (2.5 m/s) crossing the room.",
    "A kitchen with a table and four chairs, daytime, camera high near the ceiling.", "An empty-looking room with only a bed and a flowerpot, dark, camera slow.",
    "A living room with a bookcase wall, a floor lamp and an armchair, lamp light.", "A bedroom with a desk and a chair next to the window, day, camera inverted.",
    "A studio apartment 5 m by 5 m with a bed, a small table and two chairs, a door on the south wall.", "A home office with two desks facing each other and a plant between them.",
    "A living room with a TV on a low cabinet, a sofa facing it and a rug-free floor, camera circling the sofa.", "A kitchen with a stove, a fridge, a table and a window above the table.",
    "A bedroom with two beds and a wardrobe, lamp light, camera crossing the room diagonally.", "A big room 8 m by 6 m with scattered chairs and tables, camera high and fast.",
    "A dark corridor-like room 2.5 m by 8 m with a few boxes (props) along one wall.", "A living room with a window on the east wall, the sofa under it not covering the window, daytime."],
  "outdoor": [
    "A town block: one 10 m wide road along x, four cars on the lanes, six trees on the kerbs, drone flying at 25 m altitude at 15 m/s.", "A quiet street with two houses, parked cars and a rover camera at 1 m height driving along the road at 5 m/s.",
    "A park: no roads, 20 trees of different species and a few rocks, drone at 10 m altitude circling slowly.", "A crossroads with two roads, 8 cars and 4 buildings far from the roads, drone at 40 m altitude, 20 m/s.",
    "Inverted camera (mounted upside down) flying over a road with trees on both kerbs at 12 m/s.", "A fast drone at 30 m/s along a straight road, trees on the kerbs, overcast.", "A dusk scene with a single road, six cars and some poles near the kerbs, rover.",
    "A village: three buildings, two roads, many shrubs and trees, drone flying low at 5 m altitude at 8 m/s.", "A highway-like road 14 m wide with ten cars on the lanes, drone following at 30 m altitude.",
    "An open field with scattered rocks and trees, no roads, drone at 3 m altitude at 20 m/s, noon.", "A residential street, eight trees in a row on the right kerb, cars on both lanes, rover at 1.5 m.",
    "A forest-like area with 40 trees, one small road crossing it, drone at 15 m altitude.", "A small industrial area with four buildings, one road, props near the buildings, drone at 20 m altitude.",
    "A narrow 6 m road between two rows of trees, camera inverted at 2 m height moving at 6 m/s.", "A city street with five buildings set back from the road, lamp poles on the kerbs, overcast, drone at 12 m altitude."],
  "race": [
    "A simple 6-gate track on open grass, 15 m/s, gates 12 to 20 m apart, some gates high in the air.", "A fast 30 m/s track with 8 gates in a gentle curve, camera tilted up 30 degrees, overcast.", "The real A2RL 2025 lap in the hall at 20 m/s.",
    "A tight track with gates 4 to 6 m apart, 10 m/s, to practise close gates, noon.", "A slalom of 8 gates alternating left and right turns of about 45 degrees, flight line offset 0.4 m, 20 m/s.", "Climbing track: gates rise from 1 m to 4 m, 12 m/s.",
    "A dusk track with 7 gates and a drone flying 25 m/s with the camera tilted 40 degrees.", "A hall track of 10 gates with sharp 90 degree turns every second gate, 15 m/s.", "A straight line of 5 gates 25 m apart at 30 m/s, high speed test.",
    "A descending track from 5 m to 1 m height, 8 gates, 18 m/s.", "A figure of 9 gates with mixed heights between 1 and 3.5 m, 22 m/s, flight line offset -0.5 m.", "An easy training track of 4 gates, 8 m/s, 10 to 15 m apart.",
    "A long 12-gate track, 25 m/s, camera tilt 35 degrees, hall setting.", "Gates with big height differences (1 m, 4 m, 1.5 m, 4.5 m, ...), 16 m/s.", "A U-turn track: 3 gates out, a 90 degree turn twice, 3 gates back, 14 m/s."]}
spec = sys.argv[1]; outp = Path(sys.argv[2]); rep = int(sys.argv[3]) if len(sys.argv) > 3 else 2; outp.parent.mkdir(parents=True, exist_ok=True)
if spec.startswith("ollama:"): ask = ollama_ask(spec[7:])
else:
    m, _, host = spec[7:].partition("@"); ask = openai_ask(m, host or "http://localhost:8000/v1")
res = []; systems = {d: DOMAINS[d]["system"]() for d in DOMAINS}
import os
for dom, reqs in REQ.items():
    if os.environ.get("ONLY_DOMAINS") and dom not in os.environ["ONLY_DOMAINS"].split(","): continue                    # e.g. ONLY_DOMAINS=outdoor
    for i, q in enumerate(reqs):
        try: r = make(dom, q, ask, rep, systems[dom])
        except Exception as e: r = {"ok": False, "rounds": None, "errors_per_round": [[repr(e)]], "layout": None, "seconds": 0.0}
        res.append({"domain": dom, "request": q, **r}); print(dom, i, "ok" if r["ok"] else "FAIL", r["rounds"], r["seconds"], "s", flush=True)
summ = {"model": spec, "n": len(res)}
for dom in REQ:
    rr = [r for r in res if r["domain"] == dom]
    if not rr: continue                                                    # domain filtered out by ONLY_DOMAINS
    errs = collections.Counter(re.sub(r"[0-9.\-]+", "#", e)[:60] for r in rr if not r["ok"] for e in (r["errors_per_round"][-1] if r["errors_per_round"] else []))
    summ[dom] = {"n": len(rr), "first_try": sum(r["ok"] and r["rounds"] == 0 for r in rr), "valid_after_repair": sum(bool(r["ok"]) for r in rr), "mean_s": round(sum(r["seconds"] for r in rr) / len(rr), 1), "top_errors": errs.most_common(4)}
summ["valid_total"] = sum(bool(r["ok"]) for r in res); json.dump({"summary": summ, "results": res}, open(outp, "w"), indent=1); print("SUMMARY", json.dumps(summ, indent=1))
