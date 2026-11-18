"""Photo -> 3D asset (prototype). One image in, a GLB + USD out with a REAL-WORLD size, standing on z=0, plus a meta file the asset registry can read.
The generator only gives the shape and colour; the class tag and the size come from us (TAG_SIZE_M), never from the model (same rule as the layouts).
Runs in the separate env /home/haktan/Projects/i23d (TripoSR, MIT licence; code is untrusted third party, kept out of this project):
  PYTHONPATH=/home/haktan/Projects/i23d/shim:/home/haktan/Projects/i23d/triposr /home/haktan/Projects/i23d/.venv/bin/python scripts/image_to_asset.py <image> <tag> <name> [--size_m X] [--res 256] [--out assets/generated]
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch, trimesh
from PIL import Image
import importlib.util                      # load the data-only module directly: `import pfm` would pull in timm, which this env does not have
_sp = importlib.util.spec_from_file_location("pfm_assets", Path(__file__).resolve().parents[1] / "src/pfm/assets.py"); _m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m); TAG_SIZE_M = _m.TAG_SIZE_M

ap = argparse.ArgumentParser(); ap.add_argument("image"); ap.add_argument("tag"); ap.add_argument("name")
ap.add_argument("--size_m", type=float); ap.add_argument("--res", type=int, default=256); ap.add_argument("--out", default="assets/generated"); ap.add_argument("--no_rembg", action="store_true")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no_upright", action="store_true"); ap.add_argument("--backend", choices=["hunyuan", "triposr"], default="hunyuan"); ap.add_argument("--max_faces", type=int, default=60000)
a = ap.parse_args()
size_m = a.size_m or TAG_SIZE_M[a.tag]                                   # unknown tag and no --size_m: KeyError on purpose
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()

import io, contextlib                                                  # noqa: E402
img = Image.open(a.image)
if a.backend == "hunyuan":
    # Hunyuan3D-2mini turbo (shape only, 0.6B, ~4.4 GB on the GPU). Licence: Tencent Hunyuan 3D 2.0 Community Licence (territory limits: not EU/UK/South Korea; read it before any release).
    sys.path.insert(0, os.environ.get("I23D", "/home/haktan/Projects/i23d") + "/hy3d")
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer
    from hy3dgen.rembg import BackgroundRemover
    rgba = img.convert("RGBA"); has_alpha = np.asarray(rgba)[..., 3].min() < 250
    cut = rgba if (has_alpha or a.no_rembg) else BackgroundRemover()(img.convert("RGB"))
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini-turbo", use_safetensors=True, variant="fp16", device="cuda")
    with torch.no_grad():
        hm = pipe(image=cut, num_inference_steps=5, octree_resolution=256, num_chunks=20000, generator=torch.manual_seed(a.seed))[0]
    hm = FaceReducer()(hm, max_facenum=a.max_faces)                       # 400-500k faces raw: too heavy for scenes
    fg = np.asarray(cut)[..., :3][np.asarray(cut)[..., 3] > 128]; base = np.median(fg, axis=0).astype(np.uint8) if len(fg) else np.array([160] * 3, np.uint8)   # shape-only model: one flat colour (median of the object in the photo), real texture needs the paint model (16+ GB GPU)
    mesh = trimesh.Trimesh(hm.vertices, hm.faces, vertex_colors=np.tile(np.append(base, 255), (len(hm.vertices), 1)), process=False)
    raw_up_y = True; gen_name = "Hunyuan3D-2mini-turbo (shape only, flat colour)"
else:
    from tsr.system import TSR                                               # noqa: E402
    from tsr.utils import remove_background, resize_foreground                # noqa: E402
    if not a.no_rembg and not (img.mode == "RGBA" and np.asarray(img)[..., 3].min() < 250):    # already cut out? keep it
        import rembg; img = remove_background(img.convert("RGB"), rembg.new_session())
    img = resize_foreground(img, 0.85); x = np.asarray(img.convert("RGBA")).astype(np.float32) / 255.0
    x = x[..., :3] * x[..., 3:4] + (1 - x[..., 3:4]) * 0.5; img = Image.fromarray((x * 255).astype(np.uint8))
    dev = "cuda"; model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt"); model.renderer.set_chunk_size(8192); model.to(dev)
    with torch.no_grad():
        codes = model([img], device=dev); mesh = model.extract_mesh(codes, True, resolution=a.res)[0]

    raw_up_y = False; gen_name = "TripoSR (stabilityai/TripoSR, MIT)"
peak = torch.cuda.max_memory_allocated() / 2**30

# TripoSR raw frame has X as UP (checked by rendering the chair: seat/legs line up along X). Rotate +90 deg about Z (x,y,z)->(-y,x,z): X becomes Y-up, a proper rotation (a plain axis swap would mirror the mesh).
# Then scale the longest extent to the real size, stand on the floor (min Y = 0), centre in the horizontal plane.
v = np.asarray(mesh.vertices, np.float64); v = v if raw_up_y else v[:, [1, 0, 2]] * np.array([-1, 1, 1])
_sp = importlib.util.spec_from_file_location("pfm_upright", Path(__file__).resolve().parents[1] / "src/pfm/mesh_upright.py"); _u = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_u)
upright = a.tag in _u.FLAT_TAGS and not a.no_upright
if upright:
    v = _u.upright_flat(v)                                                 # vehicles/beds/sofas/tables: gravity from the shape, not from the photo's viewpoint
ext = v.max(0) - v.min(0); v = (v - v.min(0)) * (size_m / ext.max()); v[:, [0, 2]] -= (v[:, [0, 2]].max(0) + v[:, [0, 2]].min(0)) / 2
mesh = trimesh.Trimesh(v, mesh.faces, vertex_colors=mesh.visual.vertex_colors, process=False)
mesh.export(out / f"{a.name}.glb")

from pxr import Usd, UsdGeom, Gf, Vt, Sdf                               # noqa: E402  (usd-core)
st = Usd.Stage.CreateNew(str(out / f"{a.name}.usd")); UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.y); UsdGeom.SetStageMetersPerUnit(st, 1.0)
m = UsdGeom.Mesh.Define(st, f"/{a.name}"); st.SetDefaultPrim(m.GetPrim())
m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, np.float32))); m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mesh.faces)))
m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(mesh.faces, np.int32).reshape(-1))); m.CreateSubdivisionSchemeAttr("none")
pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
pv.Set(Vt.Vec3fArray.FromNumpy((np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0).astype(np.float32))); st.GetRootLayer().Save()

meta = {"name": a.name, "tag": a.tag, "size_m_longest": size_m, "extent_m": [round(float(e), 3) for e in mesh.extents], "faces": int(len(mesh.faces)), "watertight": bool(mesh.is_watertight),
        "source_image": str(a.image), "generator": gen_name, "peak_gpu_gb": round(peak, 2), "seconds": round(time.time() - t0, 1), "seed": a.seed, "upright_pca": bool(upright), "up_axis": "Y"}
json.dump(meta, open(out / f"{a.name}.json", "w"), indent=1); print(json.dumps(meta))
