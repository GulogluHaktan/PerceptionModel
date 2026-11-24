"""Uniform weight soup: average the plain state dicts (final.pt) of several runs that share one initialisation (e.g. fine-tunes made with init_from).
Usage: python scripts/soup_avg.py OUT.pt IN1.pt IN2.pt [...]
Float tensors are averaged in float32 and cast back; integer buffers (e.g. num_batches_tracked) are taken from the first input.
Averaging runs that do NOT share an initialisation breaks the model (Update 127)."""
import sys
import torch

out, ins = sys.argv[1], sys.argv[2:]
assert len(ins) >= 2, "need at least two inputs"
sds = [torch.load(p, map_location="cpu") for p in ins]
keys = list(sds[0])
for p, sd in zip(ins[1:], sds[1:]):
    assert list(sd) == keys, f"{p}: different keys than {ins[0]}"
avg = {}
for k in keys:
    t = sds[0][k]
    if t.is_floating_point():
        assert all(sd[k].shape == t.shape for sd in sds), f"shape mismatch at {k}"
        avg[k] = (sum(sd[k].float() for sd in sds) / len(sds)).to(t.dtype)
    else:
        avg[k] = t.clone()
torch.save(avg, out)
d = max((avg[k].float() - sds[0][k].float()).abs().max().item() for k in keys if sds[0][k].is_floating_point())
print(f"soup of {len(ins)} -> {out} ({len(keys)} tensors, max |soup - first| = {d:.3g})")
