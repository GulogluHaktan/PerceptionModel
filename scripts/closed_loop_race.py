"""Closed-loop gate passing in simulation: does the perception model, in the loop, get a simple drone through the gates?  SIMULATION RESULT: rendered scenes like the training data, a point-mass drone,
perfect knowledge of the drone's own pose, no wind, no rolling shutter. It measures whether perception errors break the passage, it is NOT a real-world success rate.
Setup per track (runs/race_job/scene.json, made by scripts/layout_to_job_race.py from a layout the model never trained on):
  drone = point mass, speed SPEED m/s, acceleration limit ACC m/s^2, 60 Hz; camera looks along the horizontal heading tilted up by the layout's tilt, 130 deg, 640x480.
  the drone has a rough gate map: each gate centre is known with an error of MAPERR m (std per axis, fixed per run and seed) and its passage direction exactly; it flies to a point in front of the gate, then through.
  A gate is passed when the drone crosses its plane inside the opening with a margin of 0.25 m (|sideways|, |up| < inner/2 - 0.25); crossing inside the frame but outside that = hit; farther out = missed.
Variants: oracle (true gate centres, the controller's upper bound, no rendering), map (rough map only, no camera, no rendering), percept (rough map corrected by the model: every frame, each predicted gate with
score > 0.5 goes through PnP (1.5 m square) -> world position; a measurement within GATE m of the current estimate of the target gate and closer than RANGE m updates that estimate), and percept_delay
(the same with the measurement used one frame late = about 17 ms of latency). DEGRADE=dark|blur|darkblur degrades the camera image before the model (crude: quarter exposure + noise; box blur growing with speed). percept needs sim/isaac_race_loop.py running on the same scene.
Usage: PFM_GATE_HEAD=dense PFM_GDET_STRIDE=4 CKPT=runs/x KIND=hybrid python scripts/closed_loop_race.py <variant> [seed] -> prints one JSON line."""
import json, math, os, sys, time
from pathlib import Path
import numpy as np
VAR = sys.argv[1]; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0; JOB = Path("runs/race_job"); LOOP = JOB / "loop"; S = json.load(open(JOB / "scene.json")); sc = S["scene"]; cc = sc["camera"]
SPEED = float(os.environ.get("SPEED", "8")); ACC = float(os.environ.get("ACC", "20")); MAPERR = float(os.environ.get("MAPERR", "1.0")); RANGE = float(os.environ.get("RANGE", "12")); GATE = float(os.environ.get("GATE", "3.0")); DT = 1 / 60; DEG = os.environ.get("DEGRADE", ""); drng = np.random.default_rng(123)
I = sc["inner"]; O = sc["outer"]; TILT = math.radians(cc["tilt_deg"]); rng = np.random.default_rng(SEED)
P = np.array([p["pos"] for p in cc["poses"]], float); gates = []
for g in sc["gates"]:      # passage axis, signed along the direction of travel of the reference path
    y = math.radians(g["yaw_deg"]); a = np.array([math.cos(y), math.sin(y), 0.0]); c = np.array(g["at"], float); k = int(np.argmin(np.linalg.norm(P - c, axis=1))); v = P[min(k + 3, len(P) - 1)] - P[max(k - 3, 0)]
    if v @ a < 0: a = -a
    gates.append({"c": c, "a": a, "r": np.array([a[1], -a[0], 0.0]), "est": c + (rng.normal(0, MAPERR, 3) if VAR != "oracle" else 0)})
percept = VAR.startswith("percept")
if percept:
    import torch, cv2
    from PIL import Image
    sys.path.insert(0, "src")
    from pfm.model import PerceptionModel
    from pfm.data import normalize
    CK = os.environ["CKPT"]; K = os.environ.get("KIND", "vit"); m = PerceptionModel(K, teacher_dims={"dino": 384}); m.load_state_dict(torch.load(f"{CK}/{K}/final.pt", map_location="cpu")); m = m.cuda().eval()
    t0 = time.time()
    while not (LOOP / "ready").exists():
        time.sleep(0.5)
        if time.time() - t0 > 900: raise SystemExit("renderer not ready")
    (LOOP / "reset").write_text("")
    while (LOOP / "reset").exists(): time.sleep(0.01)
    cam = json.loads((LOOP / "ready").read_text()); FX, CW, CH = cam["fx"], cam["width"], cam["height"]; KM = np.array([[FX, 0, CW / 2], [0, FX, CH / 2], [0, 0, 1.0]])
    OBJ = np.array([[-I / 2, I / 2, 0], [I / 2, I / 2, 0], [I / 2, -I / 2, 0], [-I / 2, -I / 2, 0]], np.float64)
def cam_axes(head):
    f = math.cos(TILT) * head + math.sin(TILT) * np.array([0, 0, 1.0]); r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f); return f, r, u
pos = P[0].copy(); head = gates[0]["c"] - pos; head[2] = 0; head /= np.linalg.norm(head); vel = SPEED * head; gi = 0; res = []; traj = []; k = 0; n_meas = 0; pending = None; est_err = []; t_sim = 0.0
while gi < len(gates) and t_sim < 120:
    g = gates[gi]
    if percept:
        f, r, u = cam_axes(head); tmp = LOOP / f"req_{k}.tmp"; tmp.write_text(json.dumps({"pos": pos.tolist(), "forward": f.tolist(), "up": u.tolist()})); tmp.rename(LOOP / f"req_{k}.json"); t0 = time.time()
        while not (LOOP / f"ack_{k}").exists():
            time.sleep(0.001)
            if time.time() - t0 > 120: raise SystemExit("renderer stopped answering")
        rgb = np.load(LOOP / f"rgb_{k}.npy"); (LOOP / f"rgb_{k}.npy").unlink(); (LOOP / f"ack_{k}").unlink(); k += 1
        if "blur" in DEG: L = int(np.clip(SPEED * 0.6, 3, 25)); rgb = cv2.blur(rgb, (L, L // 2 + 1))                       # crude motion blur that grows with speed (box filter, not the pfm.effects model)
        if "dark" in DEG: rgb = np.clip(rgb.astype(np.float32) * 0.25 + drng.normal(0, 6, rgb.shape), 0, 255).astype(np.uint8)   # quarter exposure + sensor noise
        x = torch.from_numpy(np.asarray(Image.fromarray(rgb).resize((448, 448), Image.BICUBIC)).copy()).permute(2, 0, 1).float()[None] / 255
        with torch.no_grad(): sl = m(normalize(x.cuda()), with_teacher_heads=False)["slots"].float().cpu()[0]
        meas = []
        for q in range(sl.shape[0]):
            if sl[q, 0].sigmoid() <= 0.5: continue
            ok, rv, tv = cv2.solvePnP(OBJ, (sl[q, 1:].view(4, 2).numpy() * [CW, CH]).astype(np.float64), KM, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok and tv[2, 0] > 0.2: meas.append(pos + tv[0, 0] * r - tv[1, 0] * u + tv[2, 0] * f)      # OpenCV camera: x right, y down, z forward
        if os.environ.get("SNAP") and (k - 1) % int(os.environ.get("SNAP_EVERY", "45")) == 0:      # camera frame with the predicted gates drawn, for pictures
            from PIL import ImageDraw
            im = Image.fromarray(rgb); d = ImageDraw.Draw(im)
            for q in range(sl.shape[0]):
                if sl[q, 0].sigmoid() > 0.5: pts = [(float(a_) * CW, float(b_) * CH) for a_, b_ in sl[q, 1:].view(4, 2).numpy()]; d.line(pts + [pts[0]], fill=(0, 255, 0), width=3)
            os.makedirs(os.environ["SNAP"], exist_ok=True); im.save(f"{os.environ['SNAP']}/f{k:04d}_gate{gi}.jpg", quality=88)
        use = pending if VAR == "percept_delay" else meas; pending = meas
        for w in (use or []):
            d = np.linalg.norm(w - pos)
            if d < RANGE and np.linalg.norm(w - g["est"]) < GATE: g["est"] = g["est"] + (0.5 if d < 6 else 0.25) * (w - g["est"]); n_meas += 1
    # guidance: aim at a point in front of the gate that moves onto the gate as the drone gets close, then straight through
    along = (g["est"] - pos) @ g["a"]; tgt = g["est"] - g["a"] * float(np.clip(0.5 * along, 0, 4)) if along > 0.3 else pos + g["a"] * 2
    dv = tgt - pos; vdes = SPEED * dv / max(np.linalg.norm(dv), 1e-6); acc = (vdes - vel) / (3 * DT); na = np.linalg.norm(acc)
    if na > ACC: acc *= ACC / na
    new = pos + vel * DT + 0.5 * acc * DT * DT; vel = vel + acc * DT; hv = vel.copy(); hv[2] = 0
    if np.linalg.norm(hv) > 0.5: head = hv / np.linalg.norm(hv)
    s0 = (pos - g["c"]) @ g["a"]; s1 = (new - g["c"]) @ g["a"]
    if s0 < 0 <= s1:      # crossed the true gate plane
        x_ = pos + (new - pos) * (-s0 / (s1 - s0)) - g["c"]; lat, up_ = abs(x_ @ g["r"]), abs(x_[2]); est_err.append(float(np.linalg.norm(g["est"] - g["c"])))
        res.append("passed" if max(lat, up_) < I / 2 - 0.25 else "hit" if max(lat, up_) < O / 2 else "missed"); gi += 1
    elif s1 > 6 and s0 >= 0: res.append("lost"); est_err.append(float(np.linalg.norm(g["est"] - g["c"]))); gi += 1      # flew past without crossing (should not happen)
    pos = new; t_sim += DT; traj.append(pos.tolist())
if percept and os.environ.get("KEEP_RENDERER") != "1": (LOOP / "stop").write_text("")
res += ["not_reached"] * (len(gates) - len(res))
if os.environ.get("TRAJ"): json.dump({"traj": traj, "gates": [{"c": g["c"].tolist(), "a": g["a"].tolist(), "est": g["est"].tolist()} for g in gates], "per_gate": res, "variant": VAR, "inner": I, "outer": O}, open(os.environ["TRAJ"], "w"))   # path for plots
print(json.dumps({"track": S["name"], "variant": VAR, "seed": SEED, "speed": SPEED, "map_err_m": MAPERR, "degrade": DEG, "gates": len(gates), "passed": res.count("passed"), "hit": res.count("hit"), "missed": res.count("missed") + res.count("lost") + res.count("not_reached"),
                  "gate_estimate_error_at_crossing_m_median": round(float(np.median(est_err)), 3) if est_err else None, "measurements_used": n_meas, "frames": k, "sim_seconds": round(t_sim, 1), "per_gate": res}))
