"""Validity test of a local LLM for layouts: N requests -> make_layout (schema-constrained, resolver, repair rounds). Usage: python scripts/llm_layout_test.py <ollama model> <out.json> [max_repairs].
Reports first-answer validity, validity after repairs, rounds and seconds per layout. Layouts that pass are stored with their request."""
import json, sys, time
sys.path.insert(0, "src")
from pfm.layout_llm import make_layout, ollama_ask, system_prompt
REQ = ["A small living room, daytime, camera walking slowly from the door side to the sofa.", "A bedroom at night with the lamp on, camera flying low at 1 m height.",
       "An office with a desk, a chair and a bookcase, camera inverted (mounted upside down), moving 2 m/s.", "A kitchen with a fridge and a stove, dark lighting.",
       "A cluttered living room with at least 6 items, one TV standing on a table, daytime.", "A narrow bedroom 3 m by 5 m with a wardrobe and a bed, camera looking at the window.",
       "A large living room 7 m by 6 m with two sofas, a table between them and plants in the corners.", "An office with a sofa in the corner, lamp light, camera fast (2.5 m/s) crossing the room.",
       "A kitchen with a table and four chairs, daytime, camera high near the ceiling.", "An empty-looking room with only a bed and a flowerpot, dark, camera slow.",
       "A living room with a bookcase wall, a floor lamp and an armchair, lamp light.", "A bedroom with a desk and a chair next to the window, day, camera inverted."]
model, outp = sys.argv[1], sys.argv[2]; rep = int(sys.argv[3]) if len(sys.argv) > 3 else 2
ask = ollama_ask(model); system = system_prompt(); res = []
for i, q in enumerate(REQ):
    t = time.time()
    try:
        r = make_layout(q, ask, max_repairs=rep, system=system)
    except Exception as e:
        r = {"ok": False, "rounds": None, "errors_per_round": [[repr(e)]], "layout": None, "seconds": round(time.time() - t, 1)}
    res.append({"request": q, **{k: r[k] for k in ("ok", "rounds", "seconds", "errors_per_round", "layout")}})
    print(i, "ok" if r["ok"] else "FAIL", "rounds", r["rounds"], r["seconds"], "s |", (r["errors_per_round"][0][:2] if r["errors_per_round"] else ""), flush=True)
n = len(res); first = sum(1 for r in res if r["ok"] and r["rounds"] == 0); ok = sum(1 for r in res if r["ok"])
summ = {"model": model, "n": n, "first_try_valid": first, "valid_after_repair": ok, "mean_s": round(sum(r["seconds"] for r in res) / n, 1)}
json.dump({"summary": summ, "results": res}, open(outp, "w"), indent=1); print("SUMMARY", json.dumps(summ), flush=True)
