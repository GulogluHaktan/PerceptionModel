"""Resolve an LLM-style layout JSON (rules in src/pfm.layout_race.py) and write runs/race_job/scene.json for sim/isaac_race.py. Usage: python scripts/layout_to_job.py <layout.json> <name>. Prints the errors and exits 1 if a rule fails."""
import importlib.util, json, sys, types
sys.path.insert(0, "src"); pkg = types.ModuleType("pfm"); pkg.__path__ = ["src/pfm"]; sys.modules["pfm"] = pkg                      # no torch: layout and assets are data-only modules
for n in ("layout_race",):
    sp = importlib.util.spec_from_file_location(f"pfm.{n}", f"src/pfm/{n}.py"); m = importlib.util.module_from_spec(sp); sys.modules[f"pfm.{n}"] = m; sp.loader.exec_module(m)
L = sys.modules["pfm.layout_race"]; r = L.resolve(json.load(open(sys.argv[1])))
if not r["ok"]:
    print("\n".join(r["errors"])); sys.exit(1)
json.dump({"name": sys.argv[2], "scene": r["scene"]}, open("runs/race_job/scene.json", "w")); print("resolved", sys.argv[2], len(r["scene"]["gates"]), "gates,", len(r["scene"]["camera"]["poses"]), "frames")
