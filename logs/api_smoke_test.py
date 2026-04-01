import json
import requests
from pathlib import Path

base = 'http://127.0.0.1:8000'
results = []

def add(name, ok, status=None, detail=None):
    results.append({'endpoint': name, 'ok': ok, 'status': status, 'detail': detail})

try:
    r = requests.get(base + '/health', timeout=20)
    add('GET /health', r.status_code == 200, r.status_code, r.json())
except Exception as exc:
    add('GET /health', False, None, str(exc))

for ep in ['/model-status', '/metrics', '/db-status', '/prediction-history?limit=5', '/training-history?limit=5']:
    try:
        r = requests.get(base + ep, timeout=20)
        payload = r.json() if 'application/json' in r.headers.get('content-type', '') else r.text[:200]
        add('GET ' + ep, r.status_code == 200, r.status_code, payload)
    except Exception as exc:
        add('GET ' + ep, False, None, str(exc))

try:
    sample = None
    for d in Path('data/test').glob('*'):
        if d.is_dir():
            imgs = list(d.glob('*.jpg')) + list(d.glob('*.jpeg')) + list(d.glob('*.png'))
            if imgs:
                sample = imgs[0]
                break
    if sample is None:
        raise RuntimeError('No test image found in data/test')
    with open(sample, 'rb') as fh:
        r = requests.post(base + '/predict', files={'file': (sample.name, fh, 'image/jpeg')}, timeout=60)
    add('POST /predict', r.status_code == 200, r.status_code, r.json())
except Exception as exc:
    add('POST /predict', False, None, str(exc))

try:
    sample = next(Path('data/test/Apple___Cedar_apple_rust').glob('*'))
    with open(sample, 'rb') as fh:
        files = [('files', (sample.name, fh, 'image/jpeg'))]
        r = requests.post(base + '/upload-data', files=files, data={'class_name': 'Apple___Cedar_apple_rust'}, timeout=60)
    add('POST /upload-data', r.status_code == 200, r.status_code, r.json())
except Exception as exc:
    add('POST /upload-data', False, None, str(exc))

# Connectivity check for retrain route without running long training.
try:
    r = requests.get(base + '/openapi.json', timeout=20)
    route_exists = False
    if r.status_code == 200:
        spec = r.json()
        route_exists = '/retrain' in spec.get('paths', {})
    add('POST /retrain (route exists)', route_exists, r.status_code, {'route_exists': route_exists})
except Exception as exc:
    add('POST /retrain (route exists)', False, None, str(exc))

print(json.dumps({'all_passed': all(x['ok'] for x in results), 'results': results}, indent=2))
