"""Thermal soak with the real label pipeline (Grounding DINO + SAM2) looping over images: logs temperatures, GPU clock, fan every 5 s.
Usage: python scripts/orin_label_soak.py <image_dir> <out.jsonl> [max_minutes=15] [abort_c=82]. Stops at max_minutes or when the hottest zone reaches abort_c (safety); prints SOAKDONE.
Tells whether a new enclosure keeps the module in a safe band; it does not decide what 'safe' is (throttling starts far above 70 C on Orin NX, check the datasheet)."""
import glob, json, sys, threading, time
from pathlib import Path
from PIL import Image
sys.path.insert(0, "src")
import pfm.labeler as L

imgs = sorted(Path(sys.argv[1]).glob("*.png")); out = sys.argv[2]; max_s = 60 * float(sys.argv[3]) if len(sys.argv) > 3 else 900; abort_c = float(sys.argv[4]) if len(sys.argv) > 4 else 82.0
def temps():
    d = {}
    for z in glob.glob("/sys/class/thermal/thermal_zone*"):
        try: d[open(z + "/type").read().strip()] = int(open(z + "/temp").read()) / 1000
        except OSError: pass
    return d
def rd(p):
    try: return open(p).read().strip()
    except OSError: return None
stop = threading.Event(); log = open(out, "w"); t0 = time.time(); hot = {"max": 0.0, "aborted": False}
def monitor():
    while not stop.is_set():
        t = temps(); m = max(t.values()) if t else 0.0; hot["max"] = max(hot["max"], m)
        gpu = glob.glob("/sys/devices/platform/*gpu*/devfreq/*/cur_freq") + glob.glob("/sys/class/devfreq/*gpu*/cur_freq")
        log.write(json.dumps({"t": round(time.time() - t0, 1), "tj": t.get("tj-thermal"), "gpu": t.get("gpu-thermal"), "cpu": t.get("cpu-thermal"), "max": m,
                              "gpu_hz": rd(gpu[0]) if gpu else None, "fan_pwm": rd(next(iter(glob.glob("/sys/devices/platform/pwm-fan/hwmon/*/pwm1")), "/nonexistent"))}) + "\n"); log.flush()
        if m >= abort_c: hot["aborted"] = True; stop.set()
        stop.wait(5)
th = threading.Thread(target=monitor, daemon=True); th.start()
lab = L.Labeler(); n = 0; per = []
while not stop.is_set() and time.time() - t0 < max_s:
    t = time.time(); lab.label(Image.open(imgs[n % len(imgs)]).convert("RGB")); per.append(round(time.time() - t, 1)); n += 1
    print("img", n, per[-1], "s", "max C", hot["max"], flush=True)
stop.set(); th.join(); log.close()
print("SOAKDONE", json.dumps({"images": n, "seconds": round(time.time() - t0), "peak_c": hot["max"], "aborted_at_limit": hot["aborted"], "first_img_s": per[:3], "last_img_s": per[-3:]}), flush=True)
