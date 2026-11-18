"""Catalogue growth: the pure logic (no torch, no GPU) of 'the LLM sees an important object that is not in the catalogue -> picture -> Hunyuan -> new asset'.
Pipeline (scripts/grow_catalogue.py runs it, one GPU model at a time): VLM detections (closed tag list) -> instance crop (SAM2 box prompt) -> image_to_asset.py -> `gate` -> `is_duplicate` -> registry file with provenance.
Rules kept from the layouts: the model never gives a size or code, only a tag from TAG_SIZE_M and a box; size comes from the tag; a generated asset is used by the resolver only if its status is "accepted".
The gate is automatic and cheap (CPU): geometry sanity, proportions against catalogue assets with the same tag, and silhouette agreement with the crop (best of 24 views)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .assets import TAG_SIZE_M

MIN_BOX_FRAC = 0.01          # boxes under 1% of the image are too small to build from
SIL = 96                     # silhouette canvas
CONSISTENCY_MIN = 0.93      # calibrated on 8 inputs (4 clean 0.975-0.991, 4 bad crops 0.69-0.90), in-sample: vault Update 19
IOU_MIN = 0.55               # calibrated in docs notes (see vault Update 18); n is small


def detection_schema() -> dict:
    """JSON schema for constrained decoding: tag is an enum of the known size table, so an invented tag cannot come out."""
    num = {"type": "number"}
    # property order = generation order: the free-text "why" (what the object is) comes first so the tag choice among ~40 enum values is made after the model has said what it sees
    det = {"type": "object", "properties": {"why": {"type": "string"}, "tag": {"enum": sorted(TAG_SIZE_M)}, "box": {"type": "array", "items": num, "minItems": 4, "maxItems": 4},
                                            "importance": num}, "required": ["why", "tag", "box", "importance"], "additionalProperties": False}
    return {"type": "object", "properties": {"objects": {"type": "array", "items": det, "maxItems": 8}}, "required": ["objects"], "additionalProperties": False}


def detect_prompt(known_tags: list[str]) -> str:
    return ("You look at one camera frame for a perception-training simulator. List objects that matter for navigation or for what the camera sees (furniture, vehicles, poles, obstacles) "
            f"and that are NOT already covered by the catalogue tags: {sorted(set(known_tags))}. For each give: why (what the object is, a few words), tag (the allowed value that matches it), box [x0,y0,x1,y1] in 0-1000 relative image coordinates "
            "tightly around ONE object (never around a group), importance 0..1. Skip walls, floor, sky, tiny or heavily cut-off objects. Reply with JSON only.")


def parse_detections(raw: str | dict, known_tags=(), min_importance: float = 0.3) -> list[dict]:
    """Validate the VLM answer: unknown tags, boxes outside 0..1 / degenerate / tiny, and tags already in the catalogue are dropped (with the reason kept in 'dropped')."""
    try:
        d = json.loads(raw) if isinstance(raw, str) else raw
        objs = d["objects"]
    except Exception:
        return []
    out = []
    for o in objs:
        try:
            tag, (x0, y0, x1, y1), imp = o["tag"], [float(v) for v in o["box"]], float(o.get("importance", 0))
        except Exception:
            continue
        if tag not in TAG_SIZE_M or tag in set(known_tags) or imp < min_importance:
            continue
        if max(x0, y0, x1, y1) > 1.5 and max(x0, y0, x1, y1) <= 1000:      # Qwen-VL style 0..1000 coordinates
            x0, y0, x1, y1 = x0 / 1000, y0 / 1000, x1 / 1000, y1 / 1000
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)
        if x1 - x0 <= 0 or y1 - y0 <= 0 or (x1 - x0) * (y1 - y0) < MIN_BOX_FRAC:
            continue
        out.append({"tag": tag, "box": [x0, y0, x1, y1], "importance": imp, "why": str(o.get("why", ""))[:200]})
    return sorted(out, key=lambda o: -o["importance"])


def silhouette(verts: np.ndarray, faces: np.ndarray, yaw_deg: float, elev_deg: float, size: int = SIL) -> np.ndarray:
    """Orthographic binary silhouette of a Y-up mesh, fitted into a size x size canvas (scale-free). Vertices + face centroids are splatted, then closed with a 3x3 max filter (fast, no rasteriser)."""
    pts = np.concatenate([verts, verts[faces].mean(1)]) if len(faces) else verts
    a, e = math.radians(yaw_deg), math.radians(elev_deg)
    R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]); T = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
    p = pts @ R.T @ T.T; u, v = p[:, 0], -p[:, 1]
    span = max(u.max() - u.min(), v.max() - v.min(), 1e-9); s = (size - 6) / span
    iu = np.round((u - (u.max() + u.min()) / 2) * s + size / 2).astype(int); iv = np.round((v - (v.max() + v.min()) / 2) * s + size / 2).astype(int)
    img = np.zeros((size, size), np.uint8); ok = (iu >= 0) & (iu < size) & (iv >= 0) & (iv < size); img[iv[ok], iu[ok]] = 255
    im = Image.fromarray(img).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    return np.asarray(im) > 0


def fit_mask(mask: np.ndarray, size: int = SIL) -> np.ndarray:
    """Crop a boolean mask to its bbox and fit it into the same canvas as `silhouette` (same scale-free normalisation)."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros((size, size), bool)
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]; h, w = m.shape; s = (size - 6) / max(h, w)
    im = Image.fromarray((m * 255).astype(np.uint8)).resize((max(1, round(w * s)), max(1, round(h * s))), Image.NEAREST); out = np.zeros((size, size), np.uint8)
    y0, x0 = (size - im.size[1]) // 2, (size - im.size[0]) // 2; out[y0:y0 + im.size[1], x0:x0 + im.size[0]] = np.asarray(im); return out > 0


def best_silhouette_iou(verts, faces, mask, yaws=range(0, 360, 30), elevs=(10, 30)) -> float:
    """Max IoU between the crop mask and the mesh silhouette over 12 yaws x 2 elevations (also mirrored: a photo can show either side)."""
    tgt = fit_mask(mask); best = 0.0
    for e in elevs:
        for y in yaws:
            s = silhouette(verts, faces, y, e)
            for sm in (s, s[:, ::-1]):
                best = max(best, float((sm & tgt).sum() / max((sm | tgt).sum(), 1)))
    return best


def seed_consistency(v1, f1, v2, f2, yaws=range(0, 360, 30), elevs=(10, 30)) -> float:
    """Mean IoU between the silhouettes of two generations (different seeds, same photo) at identical views. The generator is stochastic where the photo does not decide the shape (the unseen back):
    meshes that disagree with themselves are low-confidence. Both are scale-fitted separately, so a size difference does not count."""
    ious = []
    for e in elevs:
        for y in yaws:
            a, b = silhouette(v1, f1, y, e), silhouette(v2, f2, y, e); ious.append((a & b).sum() / max((a | b).sum(), 1))
    return float(np.mean(ious))


def ratios(extents) -> list[float]:
    """Sorted extents / longest (1, mid, small): the proportion fingerprint of an object, independent of size and axis order."""
    e = sorted(float(x) for x in extents)[::-1]; return [e[1] / e[0], e[2] / e[0]]


def proportion_prior(tag: str, dims_by_name: dict, registry: dict, margin: float = 0.4, min_n: int = 2):
    """Allowed (mid, small) ratio ranges for `tag` from catalogue assets with that tag (needs >= min_n; the catalogue has only 2 per tag today, so this is a coarse range), widened by `margin`. None = no prior."""
    rs = [ratios(dims_by_name[n]) for n, a in registry.items() if a.get("tag") == tag and n in dims_by_name]
    if len(rs) < min_n:
        return None
    rs = np.array(rs); return [(float(rs[:, i].min() * (1 - margin)), float(rs[:, i].max() * (1 + margin))) for i in range(2)]


def gate(verts, faces, mask, tag, prior=None, max_faces: int = 200000, iou_min: float = IOU_MIN, alt=None, consistency_min: float | None = CONSISTENCY_MIN) -> dict:
    """Accept/reject a generated mesh. Checks: finite and non-degenerate geometry, face budget, proportions inside the tag's catalogue prior, silhouette agreement with the crop."""
    reasons, m = [], {}
    v = np.asarray(verts, float); f = np.asarray(faces)
    if not np.isfinite(v).all() or len(f) < 500:
        reasons.append("empty or non-finite mesh"); return {"ok": False, "reasons": reasons, "metrics": m}
    ext = v.max(0) - v.min(0); m["faces"] = int(len(f)); m["extents"] = [round(float(x), 3) for x in ext]
    if len(f) > max_faces:
        reasons.append(f"{len(f)} faces > {max_faces}")
    if ext.min() < 1e-4 * ext.max():
        reasons.append("flat mesh (one extent ~0)")
    else:
        r = ratios(ext); m["ratios"] = [round(x, 3) for x in r]
        if prior is not None and not all(lo <= x <= hi for x, (lo, hi) in zip(r, prior)):
            reasons.append(f"proportions {m['ratios']} outside the catalogue range for '{tag}' {[(round(a, 2), round(b, 2)) for a, b in prior]}")
    m["iou"] = round(best_silhouette_iou(v, f, mask), 3)
    if m["iou"] < iou_min:
        reasons.append(f"silhouette IoU {m['iou']} < {iou_min}")
    if alt is not None:                                              # second generation with another seed: (verts, faces)
        m["seed_consistency"] = round(seed_consistency(v, f, np.asarray(alt[0], float), np.asarray(alt[1])), 3)
        if consistency_min is not None and m["seed_consistency"] < consistency_min:
            reasons.append(f"two seeds disagree: consistency {m['seed_consistency']} < {consistency_min}")
    return {"ok": not reasons, "reasons": reasons, "metrics": m}


def dhash(img: Image.Image, n: int = 8) -> int:
    g = np.asarray(img.convert("L").resize((n + 1, n), Image.BILINEAR), np.float32); bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def is_duplicate(h: int, tag: str, registry_meta: list[dict], max_dist: int = 6) -> str | None:
    """Name of an accepted generated asset with the same tag whose source-crop hash is within `max_dist` bits, else None."""
    for g in registry_meta:
        if g.get("status") == "accepted" and g.get("tag") == tag and "crop_dhash" in g and bin(h ^ int(g["crop_dhash"], 16)).count("1") <= max_dist:
            return g["name"]
    return None


def load_accepted(gen_dir="assets/generated") -> list[dict]:
    """Generated assets the resolver may use: meta files with status == 'accepted'."""
    out = []
    for p in sorted(Path(gen_dir).glob("*.json")):
        try:
            g = json.loads(p.read_text())
        except Exception:
            continue
        if g.get("status") == "accepted" and (Path(gen_dir) / f"{g['name']}.glb").exists():
            out.append(g)
    return out
