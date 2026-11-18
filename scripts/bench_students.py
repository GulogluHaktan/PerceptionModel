import json, sys
sys.path.insert(0, "src")
from pfm import PerceptionModel
from pfm.bench import latency

for kind in ["vit", "hybrid"]:
    m = PerceptionModel(kind)
    print(kind, json.dumps({k: round(v, 2) for k, v in latency(m).items()}))
