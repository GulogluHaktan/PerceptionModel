"""Car-level street scenes written by a local LLM (Update 167): qwen3:32b (ollama) turns plain-language requests into outdoor layouts
(pfm.layout_out rules, repair loop in pfm.layout_llm_any.make); valid layouts are saved as runs/outdoor_job/layouts/car_<i>.json and
rendered with sim/run_outdoor_set.sh (Isaac Sim, exact class labels). The camera is a rover at car height, as on our vehicle.
Run: env -u PYTHONPATH .venv/bin/python scripts/llm_car_scenes.py [model=qwen3:32b]"""
import json, sys
from pathlib import Path
sys.path.insert(0, "src")
from pfm.layout_llm_any import make, ollama_ask, DOMAINS
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:32b"
REQ = [
    "A city street with one 10 m wide road, eight cars on both lanes, buildings on both sides, poles on the kerbs, rover camera at 1.2 m height driving along the road at 10 m/s, noon.",
    "A crossroads with two roads, ten cars, four buildings and street poles at the corners, rover camera at 1.3 m driving through the junction at 8 m/s, overcast.",
    "A residential street: houses on both sides, parked cars on both kerbs, trees every few metres, rover at 1.1 m driving slowly at 5 m/s, dusk.",
    "A wide 14 m avenue with twelve cars on the lanes, tall buildings, poles on both kerbs, rover at 1.5 m following traffic at 15 m/s, noon.",
    "A narrow 6 m street between buildings, three parked cars, two poles, rover at 1.2 m at 6 m/s, overcast.",
    "A suburban road lined with trees on both kerbs, four cars, one building far away, rover at 1.3 m at 12 m/s, dusk.",
    "Two parallel roads with cars, buildings between them, poles along both roads, rover at 1.2 m driving along the first road at 10 m/s, noon.",
    "A busy junction: three roads meeting, fourteen cars, five buildings, many poles, rover at 1.4 m crossing at 7 m/s, overcast.",
    "A street at dusk with six cars, street poles on the right kerb, shrubs and trees on the left kerb, rover at 1.2 m at 9 m/s.",
    "A long straight road with cars far ahead and parked cars close by, trees and poles on both kerbs, rover at 1.1 m at 20 m/s, noon.",
    "A town centre with four buildings around a crossroads, eight cars, poles at every corner, rover at 1.3 m turning at the junction at 5 m/s, dusk.",
    "A road through a park: trees and shrubs on both sides, three cars, two poles, no buildings, rover at 1.2 m at 11 m/s, overcast.",
    "A street with cars parked on both kerbs and cars driving in both lanes, buildings close to the road, rover at 1.0 m at 8 m/s, noon.",
    "A ring of roads around a block: two roads, buildings in the middle, ten cars, poles, rover at 1.4 m at 10 m/s, overcast.",
    "An industrial street: big buildings, few trees, five cars and rocks near the kerbs, rover at 1.3 m at 9 m/s, dusk.",
    "A quiet back street with two cars, many poles on both kerbs, shrubs, rover at 1.2 m at 4 m/s, noon.",
    "A main road with a side road joining it, twelve cars, trees and poles, buildings on the corners, rover at 1.5 m at 14 m/s, overcast.",
    "A street lined with palm-like trees and shops, six cars, rover at 1.2 m at 7 m/s, noon.",
    "A road leaving the town: buildings at the start, then only trees and rocks, four cars, rover at 1.3 m at 16 m/s, dusk.",
    "A parking-like street with many cars close to the camera, poles, two buildings, rover at 1.1 m at 3 m/s, overcast.",
]
SUFFIX = (" Camera: 10 frames per second, about 120 to 180 frames in total. Cars stand on the kerbs (kerb_left / kerb_right) or on the opposite lane"
          " (lane_left); the camera drives in the middle of the right lane, at least 2.5 m away from every car, pole and tree.")
REQ = [q + SUFFIX for q in REQ]
ask = ollama_ask(MODEL); system = DOMAINS["outdoor"]["system"](); ok = 0; out = Path("runs/outdoor_job/layouts")
for i, q in enumerate(REQ):
    try: r = make("outdoor", q, ask, 3, system)
    except Exception as e: r = {"ok": False, "rounds": None, "errors_per_round": [[repr(e)]], "layout": None, "seconds": 0.0}
    if r["ok"]: (out / f"car_{i:02d}.json").write_text(json.dumps(r["layout"])); ok += 1
    print(i, "ok" if r["ok"] else "FAIL", r["rounds"], round(r["seconds"], 1), "s", "" if r["ok"] else str(r["errors_per_round"][-1][:2])[:200], flush=True)
print("valid", ok, "of", len(REQ))
