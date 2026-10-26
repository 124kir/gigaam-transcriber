"""Поиск локальных моделей Ollama."""
import json
import os
import subprocess

CREATENOWINDOW = 0x08000000 if os.name == 'nt' else 0


def detect_ollama_models():
    models = []
    try:
        import requests
        r = requests.get('http://127.0.0.1:11434/api/tags', timeout=1.5)
        if r.ok:
            data = r.json()
            for item in data.get('models', []):
                name = item.get('name')
                if name: models.append(name)
            if models: return sorted(set(models))
    except Exception:
        pass
    
    # REST API fallback
    try:
        p = subprocess.run(['ollama', 'list', '--json'], capture_output=True, text=True, 
                          creationflags=CREATENOWINDOW)
        if p.returncode == 0 and p.stdout.strip():
            for line in p.stdout.splitlines():
                try:
                    obj = json.loads(line)
                    n = obj.get('name')
                    if n: models.append(n)
                except Exception:
                    continue
            if models: return sorted(set(models))
    except Exception:
        pass
    
    # CLI table fallback
    try:
        p = subprocess.run(['ollama', 'list'], capture_output=True, text=True, 
                          creationflags=CREATENOWINDOW)
        if p.returncode == 0 and p.stdout:
            for line in p.stdout.splitlines()[1:]:
                cols = [c for c in line.split() if c]
                if cols: models.append(cols[0])
            if models: return sorted(set(models))
    except Exception:
        pass
    return models
