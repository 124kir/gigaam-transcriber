"""Загрузка и сохранение настроек приложения."""
import json
import os

CONFIG_FILE = "gigaam_config.json"


def load_config():
    """Загружает сохраненные настройки из файла"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE)
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                result = {
                    # Токен больше не читаем из файла проекта: он может попасть
                    # в резервную копию или репозиторий. Задайте его один раз в
                    # окружении Windows: setx HF_TOKEN "hf_...".
                    'hf_token': os.environ.get('HF_TOKEN', ''),
                    'ollama_model': config.get('ollama_model', 'mistral:7b'),
                    'gigaam_model': config.get('gigaam_model', 'rnnt'),
                    'files': [path for path in config.get('files', []) if isinstance(path, str) and os.path.isfile(path)],
                    'diarize': bool(config.get('diarize', False)),
                    'summarize': bool(config.get('summarize', True)),
                    'export_srt': bool(config.get('export_srt', True)),
                    'export_vtt': bool(config.get('export_vtt', False)),
                    'export_json': bool(config.get('export_json', False)),
                    'export_docx': bool(config.get('export_docx', False)),
                }
                print(f"✅ Настройки загружены: модель={result['ollama_model']}, токен={'задан в окружении' if result['hf_token'] else 'пусто'}")
                return result
        except Exception as e:
            print(f"⚠️ Ошибка загрузки конфигурации: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"ℹ️ Файл конфигурации не найден: {config_path}")
    return {'hf_token': os.environ.get('HF_TOKEN', ''), 'ollama_model': 'mistral:7b'}

def save_config(hf_token: str, ollama_model: str, files=None, diarize=False,
                export_srt=True, export_vtt=False, export_json=False, export_docx=False,
                gigaam_model='rnnt', summarize=True):
    """Сохраняет настройки в файл"""
    try:
        config = {
            'ollama_model': ollama_model if ollama_model else 'mistral:7b',
            'gigaam_model': gigaam_model or 'rnnt',
            'files': list(files or [])[-50:],
            'diarize': bool(diarize),
            'summarize': bool(summarize),
            'export_srt': bool(export_srt),
            'export_vtt': bool(export_vtt),
            'export_json': bool(export_json),
            'export_docx': bool(export_docx),
        }
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"💾 Настройки сохранены: модель={config['ollama_model']}")
    except Exception as e:
        print(f"⚠️ Ошибка сохранения конфигурации: {e}")
        import traceback
        traceback.print_exc()
