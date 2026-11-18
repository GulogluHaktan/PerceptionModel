"""Like layout_to_job.py, but ALSO offers the generated assets in assets/generated/*.json with status accepted (image_to_asset.py / grow_catalogue.py) to the resolver, in memory only: assets/models/manifest.json and dims.json (the catalogue the generators and the LLM draw from) stay untouched.
Asset id = gen_<name>. Usage: python scripts/layout_to_job_gen.py <layout.json> <name>"""
import glob, importlib.util, json, sys, types
sys.path.insert(0, "src"); pkg = types.ModuleType("pfm"); pkg.__path__ = ["src/pfm"]; sys.modules["pfm"] = pkg
for n in ("assets", "layout"):
    sp = importlib.util.spec_from_file_location(f"pfm.{n}", f"src/pfm/{n}.py"); m = importlib.util.module_from_spec(sp); sys.modules[f"pfm.{n}"] = m; sp.loader.exec_module(m)
A, L = sys.modules["pfm.assets"], sys.modules["pfm.layout"]
ACC = importlib.util.spec_from_file_location("pfm.asset_growth", "src/pfm/asset_growth.py"); AG = importlib.util.module_from_spec(ACC); sys.modules["pfm.asset_growth"] = AG; ACC.loader.exec_module(AG)
for g in AG.load_accepted():                                             # only status == accepted (gate passed or manual review)
    e = A._e(f"gen_{g['name']}", "other_obstacle", f"assets/generated/{g['name']}.glb", "indoor", "glb", g["generator"], f"generated from {g['source_image']}; tag={g['tag']}"); e["tag"] = g["tag"]
    A.REGISTRY[e["name"]] = e; ex = g["extent_m"]; L.DIMS[e["name"]] = [ex[0], ex[2], ex[1]]          # glb extents are x, y(up), z -> footprint x, z and height y
r = L.resolve(json.load(open(sys.argv[1])))
if not r["ok"]:
    print("\n".join(r["errors"])); sys.exit(1)
json.dump({"name": sys.argv[2], "scene": r["scene"]}, open("runs/indoor_job/scene.json", "w")); print("resolved", sys.argv[2], len(r["scene"]["items"]), "items,", len(r["scene"]["camera"]["poses"]), "frames")
