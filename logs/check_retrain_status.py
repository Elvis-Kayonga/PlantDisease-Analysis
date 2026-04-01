import json
import requests

base = "http://127.0.0.1:8000"

m = requests.get(f"{base}/model-status", timeout=30).json()
t = requests.get(f"{base}/training-history?limit=5", timeout=30).json()
d = requests.get(f"{base}/db-status", timeout=30).json()

print(json.dumps({
    "training_in_progress": m.get("training_in_progress"),
    "last_trained": m.get("last_trained"),
    "num_classes": m.get("num_classes"),
    "training_history_count": t.get("count"),
    "training_items": t.get("items"),
    "db_training_runs": d.get("training_runs")
}, indent=2))
