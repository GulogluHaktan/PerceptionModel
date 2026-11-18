"""Asset registry (data only, no Isaac import): every placeable object maps to ONE project class, so the label comes from the asset, not from a detector.
`kind` is "usd" (referenced from NVIDIA's public Isaac asset bucket or a local file) or "procedural" (built in code, fallback while no real asset exists).
Swapping a fallback for a real asset = changing that entry only. Licences: NVIDIA Isaac/Omniverse sample assets (check the current NVIDIA licence before any release)."""
from __future__ import annotations

NV = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0"
RIVER = f"{NV}/Isaac/Environments/Outdoor/Rivermark/dsready_content/nv_content/common_assets"
TREES = f"{NV}/NVIDIA/Assets/Vegetation/Trees"
VEG = f"{NV}/NVIDIA/Assets/Vegetation"
SPECIES = ["American_Beech", "Black_Oak", "Black_Oak_Fall", "Blue_Berry_Elder", "Chinese_Juniper", "Colorado_Spruce", "Common_Apple", "Dogwood", "Douglas_Fir",
           "Eastern_Hemlock", "Elm_Sapling", "Fraxinus", "Golden_Chain", "Gray_Birch", "Gray_Birch_fall", "Hawthorn", "Honey_Locust", "Honey_Locust_Fall",
           "Italian_Cypress", "Japanese_Cherry", "Japanese_Maple", "Japanese_Maple_Fall", "Kousa_Dogwood", "Largetooth_Aspen", "Lombardy_Poplar",
           "Lombardy_Poplar_Fall", "Norway_Spruce", "Orange_Tree", "Red_Ash", "Red_Ash_Fall", "Red_Cedar", "Red_Maple", "Red_Oak", "Scarlet_Oak",
           "Scarlet_Oak_fall", "Service_Berry", "Shumard_Oak", "Shumard_Oak_Fall", "Siberian_Crab_Apple", "Sugar_Maple", "Sycamore", "White_Ash",
           "White_Pine", "Yellow_Pine"]                                                     # every species listed under Vegetation/Trees on 2026-11-11
PALMS = ["Cuban_Royal_Palm", "Fan_Palm", "Golden_Malay_Palm", "Hurricane_Palm", "Travelers_Palm", "Windmill_Palm", "Australian_Tree_Fern", "Buddha_Belly_Bamboo"]
SHRUBS = ["Acacia", "Barberry", "Boxwood", "Burning_Bush", "Cedar_Shrub", "Daphne", "Forsythia", "Fraser_Photina", "Goldflame_Spirea", "Holly", "Hydrangea",
          "Juniper", "Lilac", "Magnolia", "Oleander", "Pampas_Grass", "Pearl_Bush", "Privet", "Rhododendron", "Sweet_Mock_Orange", "Vibernum", "Yew"]
PROPS = ["bench_wrought_iron_02", "bike_rack_01", "concrete_block_02", "dumpster_blue_sm_01", "dumpster_lrg_01", "dumpster_lrg_02", "firehydrant01",
         "firehydrant04", "hydrant_03", "patio_umbrella_01", "planter_round_02", "safety_railing_01", "shopping_cart_corral_01", "table01",
         "trashcan_cylinder_01", "trashcan_square_01", "lamppost_double_01", "lamppost_single_01"]
POLES = ["pole_fxd_streetlamp_0%d" % i for i in (1, 2, 3, 4, 5, 6, 8, 9)] + ["utility_pole_30ft_01", "utility_pole_40ft_01", "utility_pole_50ft_01", "utility_pole_60ft_01",
         "tel_pole_avg_reg_50ft_01", "tel_pole_avg_tall_75ft_01", "signal_pole_avg_tall_40ft_01", "double_utility_pole_65ft", "metal_cylinderpost_12ft", "streetlamp_modern_01_30ft"]


def _e(name, cls, uri, group, kind="usd", source="NVIDIA", note=""):
    return {"name": name, "cls": cls, "kind": kind, "uri": uri, "group": group, "source": source, "note": note}


REGISTRY = {a["name"]: a for a in (
    *[_e(f"tree_{s.lower()}", "tree_pole", f"{TREES}/{s}.usd", "tree") for s in SPECIES],
    *[_e(f"palm_{s.lower()}", "tree_pole", f"{VEG}/Plant_Tropical/{s}.usd", "tree", note="tropical") for s in PALMS],
    *[_e(f"shrub_{s.lower()}", "other_obstacle", f"{VEG}/Shrub/{s}.usd", "shrub", note="low vegetation counted as obstacle, not as tree_pole") for s in SHRUBS],
    *[_e(f"rock_small_{i:02d}", "rock", f"{VEG}/Rocks/rock_small_{i:02d}.usda", "rock") for i in range(1, 16)],
    *[_e(f"prop_{n}", "other_obstacle", f"{RIVER}/props_general/{n}/{n}.usd", "prop") for n in PROPS],
    *[_e(f"pole_{n}", "tree_pole", f"{RIVER}/props_poles/{n}/{n}.usd", "pole", note="class tree_pole = trees and poles share a class (D2)") for n in POLES],
    _e("bldg_apt_complex", "building", f"{RIVER}/props_structures/apt_complex_assembly/apt_complex_assembly.usd", "building", note="assembly: check size in the asset test"),
    *[_e(f"bldg_plaza_{i:02d}", "building", f"{RIVER}/props_structures/rivermark_plaza_bldg_{i:02d}/rivermark_plaza_bldg_{i:02d}.usd", "building") for i in range(1, 9)],
    _e("bldg_library", "building", f"{RIVER}/props_structures/bldg_modern_library/bldg_modern_library.usd", "building"),
    _e("bench_curved", "other_obstacle", f"{RIVER}/props_general/bench_curved_01/bench_curved_01.usd", "prop"),
    _e("bollard", "other_obstacle", f"{RIVER}/props_general/bollard_01/bollard_01.usd", "prop"),
    _e("concrete_block", "other_obstacle", f"{RIVER}/props_general/concrete_block_01/concrete_block_01.usd", "prop"),
    _e("streetlamp", "tree_pole", f"{RIVER}/props_poles/pole_fxd_streetlamp_07/pole_fxd_streetlamp_07.usd", "pole", note="class tree_pole = trees and poles share a class (D2)"),
    _e("grass_01", "background", f"{RIVER}/props_vegetation/grass_01/grass_01.usd", "grass", note="decoration: not a label class"),
    *[_e(f"car_{n}", "car", f"assets/cars/car_{i:02d}.glb", "car", "glb", "Objaverse (CC BY, see assets/cars/ATTRIBUTION.json)", f"{n}; glTF, Y-up, rescaled to 4.5 m long")
      for i, n in ((1, "smart_fortwo"), (3, "hummer_h2"), (5, "chevrolet_c10"), (6, "mini_truck"))],
    # no NVIDIA passenger car / road mesh exists on the public bucket (checked 2026-11-10): cars come from Objaverse GLBs (CC BY, attribution required), road stays procedural
    _e("car_box", "car", "procedural:car", "car", "procedural", "code", "body + cabin boxes, ~4.5 x 1.8 x 1.5 m (fallback only)"),
    _e("road_strip", "ground_road", "procedural:road", "road", "procedural", "code", "asphalt strip with lane-marking class patches"),
    _e("rock_blob", "rock", "procedural:rock", "rock", "procedural", "code", "squashed icosphere"),
)}


# real-world longest extent (m) per fine tag: GLB sizes are arbitrary, so the spawner rescales each model to this (typical values, not exact)
TAG_SIZE_M = {"car": 4.5, "jeep": 4.0, "minivan": 4.6, "pickup_truck": 5.0, "truck": 7.5, "ambulance": 5.5, "bus": 11.0, "cab": 4.5, "police_car": 4.6, "race_car": 4.4,
              "school_bus": 10.0, "airplane": 14.0, "helicopter": 11.0, "bicycle": 1.8, "motor_scooter": 1.8, "motorcycle": 2.2, "chair": 0.9, "armchair": 0.9,
              "sofa": 2.0, "table": 1.4, "desk": 1.4, "bed": 2.0, "bookcase": 1.8, "cabinet": 1.8, "refrigerator": 1.8, "television_set": 0.8, "lamp": 1.5,
              "door": 2.0, "window": 1.2, "bathtub": 1.7, "toilet": 0.7, "sink": 0.6, "stove": 0.9, "microwave_oven": 0.5, "flowerpot": 0.5, "wardrobe": 1.9,
              "dresser": 1.0, "stool": 0.5, "ladder": 2.0}


def target_size_m(a: dict) -> float:
    return TAG_SIZE_M.get(a.get("tag", ""), 4.5 if a["cls"] == "car" else 1.0)


def _load_manifest():
    """Extra Objaverse GLBs (CC0 / CC BY) listed in assets/models/manifest.json: cars, aircraft, two-wheelers, indoor furniture.
    The fine-grained `tag` (airplane, sofa, ...) rides in the registry entry; the project class stays the coarse one (car or other_obstacle): a new class needs a new head."""
    import json
    from pathlib import Path
    mp = Path(__file__).resolve().parents[2] / "assets" / "models" / "manifest.json"
    if not mp.exists():
        return
    for name, m in json.loads(mp.read_text()).items():
        e = _e(f"{m['group']}_{name}", m["cls"], m["uri"], m["group"], "glb", f"Objaverse {m['license']} - {m['author']}", f"{m['title']}; tag={m['tag']}; glTF, rescaled on spawn")
        e["tag"] = m["tag"]
        REGISTRY[e["name"]] = e


_load_manifest()


def by_group(group: str) -> list[dict]:
    return [a for a in REGISTRY.values() if a["group"] == group]


def validate_registry():
    from .classes import CLASSES
    ok = set(CLASSES) | {"background"}
    for a in REGISTRY.values():
        if a["cls"] not in ok:
            raise ValueError(f"{a['name']}: class {a['cls']} not in project classes")
        if a["kind"] in ("usd", "glb") and not a["uri"].endswith((".usd", ".usda", ".usdc", ".glb")):
            raise ValueError(f"{a['name']}: bad uri")
    return len(REGISTRY)
