# Transcriber: GigaAM-v3 + pyannote (диаризация) + Ollama (саммари)

import os
import sys

# Если файл запустили системным Python, автоматически переключаемся на
# проектное окружение. Это позволяет использовать обычную команду
# `python main.py`, даже когда зависимости установлены только в `venv`.
_project_dir = os.path.dirname(os.path.abspath(__file__))
_venv_python = os.path.join(
    _project_dir,
    "venv",
    "Scripts" if os.name == "nt" else "bin",
    "python.exe" if os.name == "nt" else "python",
)
_running_this_file = (
    bool(sys.argv)
    and os.path.isfile(sys.argv[0])
    and os.path.normcase(os.path.abspath(sys.argv[0])) == os.path.normcase(__file__)
)
if _running_this_file and os.path.isfile(_venv_python) and os.path.normcase(sys.executable) != os.path.normcase(_venv_python):
    os.execv(_venv_python, [_venv_python, *sys.argv])

import json
import ctypes
import torch
from contextlib import contextmanager

# В Windows консоль часто остаётся в cp1251 даже после `chcp 65001`.
# Логи приложения содержат emoji, поэтому не даём выводу уронить запуск
# до создания GUI из-за UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

# Импорт GigaAM может загружать тяжёлые зависимости. Откладываем его до
# первого реального запуска: это ускоряет открытие интерфейса и collection
# тестов, которые проверяют экспорт и утилиты без ASR-модели.
gigaam = None
import logging
import threading
import datetime
import subprocess
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tempfile
import torchaudio
from typing import List, Tuple
from transformers import AutoModel
# Ленивый импорт pyannote - загружается только при необходимости
# from pyannote.audio import Pipeline
# import pyannote.core
import soundfile as sf
import warnings

from app_config import CONFIG_FILE, load_config, save_config
from exports import (format_srt_time, format_vtt_time, parse_transcript_timestamps,
                     export_srt, export_vtt, export_json, export_docx)
from ollama_utils import detect_ollama_models

# -*- coding: utf-8 -*-
TITLE = "Transcriber (GigaAM + Ollama)"
STAGES = ["GigaAM 0%", "Диаризация 25%", "GigaAM 50%", "Суммаризация 75%", "Готово 100%"]
STAGES_RU = ["GigaAM загрузка...", "Диаризация спикеров...", "GigaAM транскрипция...", "Ollama суммаризация...", "Готово!"]

ALIGNMODEL = "facebook/wav2vec2-large-xlsr-53"
PROMPTTEMPLATE = """Суммаризуй транскрипт встречи. Выдели ключевые моменты, решения, задачи. 

Транскрипт:
{TRANSCRIPT}

Краткая сводка:"""

CREATENOWINDOW = 0x08000000 if os.name == 'nt' else 0

def tsnow():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def parent_dir(path):
    return os.path.dirname(os.path.abspath(path))

def make_result_folder(audio_path):
    base = safe_result_name(audio_path)
    stamp = tsnow()
    root = parent_dir(audio_path)
    outdir = os.path.join(root, f"gigaam_{base}_{stamp}")
    ensure_dir(outdir)
    return outdir


def safe_result_name(audio_path):
    """Имя записи для папки и файлов результатов без запрещённых символов Windows."""
    import re
    base = os.path.splitext(os.path.basename(audio_path))[0]
    base = re.sub(r'[<>:"/\\|?*]+', '_', base).strip(' ._')
    return base or 'audio'

# Глобальные модели GigaAM + pyannote
GIGA_MODEL = None
DIAR_MODEL = None
GIGAAM_MODEL_NAME = 'rnnt'


@contextmanager
def _gigaam_legacy_torch_load():
    """Временно включить режим загрузки checkpoint, используемый GigaAM 0.1.0."""
    original_torch_load = torch.load

    def trusted_checkpoint_load(*args, **load_kwargs):
        # lightning/pyannote передаёт weights_only=None явно.
        if load_kwargs.get("weights_only") is None:
            load_kwargs["weights_only"] = False
        return original_torch_load(*args, **load_kwargs)

    torch.load = trusted_checkpoint_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def _load_gigaam_model(model_type, **kwargs):
    """Загрузить старый checkpoint GigaAM на новых версиях PyTorch.

    GigaAM 0.1.0 сохраняет в checkpoint служебные OmegaConf-объекты и
    вызывает torch.load без weights_only. Начиная с PyTorch 2.6 значение
    по умолчанию изменилось на True, поэтому загрузка ломается до запуска
    модели. Checkpoint берётся официальным пакетом GigaAM, поэтому здесь
    явно разрешаем старый режим только на время этой загрузки.
    """
    global gigaam
    if gigaam is None:
        from io import StringIO
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            import gigaam as gigaam_module
            gigaam = gigaam_module
        finally:
            sys.stderr = old_stderr
    with _gigaam_legacy_torch_load():
        return gigaam.load_model(model_type, **kwargs)


def _transcribe_gigaam_longform(model, audio_path):
    """Вызвать longform GigaAM вместе с совместимой загрузкой VAD checkpoint."""
    with _gigaam_legacy_torch_load():
        return model.transcribe_longform(audio_path)

# Глобальные переменные для pipeline
STOP_FLAG = False
UI_CALLBACK = None  # Функция для обновления UI

# Определяем устройство для вычислений
if torch.cuda.is_available():
    device = "cuda"
    print(f"✅ CUDA доступна: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA версия: {torch.version.cuda}")
else:
    device = "cpu"
    print("⚠️ CUDA недоступна, используется CPU")

def load_gigaam_models(hf_token):
    global GIGA_MODEL, device, GIGAAM_MODEL_NAME
    selected_model = getattr(sys.modules[__name__], 'SELECTED_GIGAAM_MODEL', GIGAAM_MODEL_NAME)
    if GIGA_MODEL is not None and GIGAAM_MODEL_NAME != selected_model:
        GIGA_MODEL = None
    if GIGA_MODEL is None:
        print(f"🔥 Официальный GigaAM на {device}...")
        # Устанавливаем токен в переменную окружения
        if hf_token and hf_token.strip():
            os.environ["HF_TOKEN"] = hf_token.strip()
        elif "HF_TOKEN" not in os.environ:
            # Пытаемся использовать токен из переменной окружения, если он там есть
            hf_token = os.environ.get("HF_TOKEN", "")
            if not hf_token:
                raise ValueError(
                    "HF_TOKEN не установлен! GigaAM требует HuggingFace токен.\n"
                    "Получите токен на https://huggingface.co/settings/tokens\n"
                    "И введите его в поле 'HF Token' в интерфейсе."
                )
        
        # Подавляем предупреждения и ошибки pyannote при загрузке
        # (GigaAM может использовать pyannote для VAD, но это не критично)
        from io import StringIO
        
        # Временно перенаправляем stderr для подавления ошибок pyannote
        old_stderr = sys.stderr
        stderr_buffer = StringIO()
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            try:
                # Подавляем ошибки pyannote при загрузке GigaAM
                sys.stderr = stderr_buffer
                try:
                    # Пытаемся загрузить модель с указанием устройства
                    # Некоторые версии gigaam поддерживают device параметр
                    try:
                        # Пытаемся загрузить модель с указанием устройства
                        if device == "cuda":
                            try:
                                GIGA_MODEL = _load_gigaam_model(selected_model, device=device)
                            except (TypeError, AttributeError, RuntimeError) as e:
                                # Если GPU не работает, пробуем загрузить на CPU
                                print(f"⚠️ Не удалось загрузить на GPU ({e}), пробуем CPU...")
                                GIGA_MODEL = _load_gigaam_model(selected_model, device="cpu")
                                device = "cpu"  # Переключаемся на CPU
                                print("ℹ️ Модель загружена на CPU")
                        else:
                            GIGA_MODEL = _load_gigaam_model(selected_model, device=device)
                    except Exception as e:
                        # Последняя попытка - загрузить без указания устройства
                        print(f"⚠️ Ошибка при загрузке с device: {e}")
                        print("   Пробуем загрузить без указания устройства...")
                        GIGA_MODEL = _load_gigaam_model(selected_model)
                        print("ℹ️ Модель загружена, GigaAM сам управляет устройством")
                finally:
                    # Восстанавливаем stderr
                    sys.stderr = old_stderr
                
                # Проверяем что модель загружена
                if GIGA_MODEL is None:
                    raise RuntimeError("Не удалось загрузить GigaAM модель!")
                GIGAAM_MODEL_NAME = selected_model
                
                # Проверяем, что модель действительно на нужном устройстве
                if device == "cuda" and GIGA_MODEL is not None:
                    try:
                        # Пытаемся проверить параметры модели
                        if hasattr(GIGA_MODEL, 'model') and GIGA_MODEL.model is not None:
                            try:
                                for param in list(GIGA_MODEL.model.parameters())[:1]:
                                    if param.device.type != "cuda":
                                        print(f"⚠️ Модель не на GPU (параметры на {param.device})")
                                    else:
                                        print(f"✅ Модель успешно на GPU")
                                    break
                            except Exception:
                                pass
                    except Exception:
                        pass
                        
            except Exception as e:
                # Восстанавливаем stderr перед выводом ошибки
                sys.stderr = old_stderr
                error_msg = str(e)
                if "HF_TOKEN" in error_msg or "token" in error_msg.lower() or "auth" in error_msg.lower():
                    raise ValueError(
                        f"Ошибка аутентификации GigaAM: {error_msg}\n\n"
                        "GigaAM требует HuggingFace токен для загрузки модели.\n"
                        "Получите токен на https://huggingface.co/settings/tokens\n"
                        "И введите его в поле 'HF Token' в интерфейсе."
                    )
                else:
                    print(f"❌ Ошибка загрузки GigaAM: {error_msg}")
                    raise RuntimeError(f"Ошибка загрузки GigaAM: {error_msg}")
        print(f"✅ GigaAM готов на {device}!")

def load_diarization_model(hf_token):
    global DIAR_MODEL, device
    if DIAR_MODEL is None:
        if not hf_token or hf_token.strip() == "":
            raise ValueError("Для использования диаризации необходим HuggingFace токен! Получите его на https://huggingface.co/settings/tokens")

        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise ImportError("pyannote.audio не установлен. Установите: pip install pyannote.audio")

        print(f"🎤 Загрузка pyannote диаризации на {device}...")
        os.environ["HF_TOKEN"] = hf_token

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            try:
                # Загружаем пайплайн
                # pyannote 3.1 также содержит старый checkpoint и вызывает
                # torch.load без weights_only, как и внутренний VAD GigaAM.
                with _gigaam_legacy_torch_load():
                    DIAR_MODEL = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=hf_token
                    )
                
                # ПЕРЕМЕЩАЕМ НА GPU ЯВНО
                if device == "cuda":
                    try:
                        DIAR_MODEL.to(torch.device("cuda"))
                        print("✅ Pyannote модель перемещена на GPU!")
                        
                        # Проверяем, что параметры модели на GPU
                        for param in list(DIAR_MODEL.parameters())[:1]:  # Только первый параметр для проверки
                            if param.device.type == "cuda":
                                print("✅ Pyannote параметры находятся на GPU")
                                break
                            else:
                                print(f"⚠️ Pyannote параметры на {param.device}, а не на GPU")
                    except Exception as e:
                        print(f"⚠️ Не удалось переместить pyannote на GPU: {e}")
                        print("   Продолжаем на CPU...")
                        
            except Exception as e:
                error_msg = str(e)
                if "token" in error_msg.lower() or "auth" in error_msg.lower():
                    raise ValueError(f"Ошибка аутентификации HuggingFace. Проверьте токен: {error_msg}")
                else:
                    raise RuntimeError(f"Ошибка загрузки диаризации: {error_msg}")
        
        print(f"✅ Диаризация готова на {device}!")

def run_gigaam_simple(audio_path: str, logfp: str, hf_token: str = "") -> str:
    global device, GIGA_MODEL
    load_gigaam_models(hf_token)
    
    # Проверяем что модель загружена
    if GIGA_MODEL is None:
        raise RuntimeError("GigaAM модель не загружена!")
    
    with open(logfp, 'a', encoding='utf-8') as flog:
        flog.write(f"Device: {device}\n")
        flog.write(f"GIGA_MODEL type: {type(GIGA_MODEL)}\n")
        flog.write(f"GIGA_MODEL is None: {GIGA_MODEL is None}\n")
        try:
            # Подавляем ошибки pyannote при транскрипции (GigaAM может использовать pyannote для VAD)
            old_stderr = sys.stderr
            from io import StringIO
            stderr_buffer = StringIO()
            try:
                sys.stderr = stderr_buffer
                # ✅ LONGFORM для любых длин!
                flog.write("Вызываю transcribe_longform...\n")
                # Внутренний VAD GigaAM загружает pyannote checkpoint
                # через старый torch.load без weights_only.
                with _gigaam_legacy_torch_load():
                    result = GIGA_MODEL.transcribe_longform(audio_path)
                flog.write(f"transcribe_longform завершен, получено {len(result) if result else 0} сегментов\n")
            except (AttributeError, RuntimeError) as e:
                # Восстанавливаем stderr для вывода ошибки
                sys.stderr = old_stderr
                error_msg = str(e)
                if "'NoneType' object has no attribute 'to'" in error_msg or "NoneType" in error_msg:
                    flog.write(f"❌ Ошибка GPU: {error_msg}\n")
                    flog.write("Проблема: GigaAM не может работать с GPU, переключаемся на CPU...\n")
                    # Пробуем перезагрузить модель на CPU
                    # device и GIGA_MODEL уже объявлены как global в начале функции
                    old_device = device
                    device = "cpu"
                    try:
                        # Перезагружаем модель на CPU
                        GIGA_MODEL = None  # Сбрасываем модель
                        load_gigaam_models(hf_token)
                        flog.write("✅ Модель перезагружена на CPU\n")
                        # Пробуем транскрипцию снова на CPU
                        result = _transcribe_gigaam_longform(GIGA_MODEL, audio_path)
                        flog.write(f"✅ Транскрипция на CPU успешна, получено {len(result) if result else 0} сегментов\n")
                    except Exception as e2:
                        flog.write(f"❌ Ошибка даже на CPU: {e2}\n")
                        raise RuntimeError(
                            f"Ошибка GigaAM: {error_msg}\n"
                            f"Попытка использовать CPU также не удалась: {e2}"
                        )
                else:
                    raise
            finally:
                sys.stderr = old_stderr
            
            transcript = []
            for utterance in result:
                text = utterance["transcription"]
                start, end = utterance["boundaries"]
                transcript.append(f"[{start:.1f}-{end:.1f}s] {text}")
            
            full_text = "\n".join(transcript)
            flog.write(f"✅ GigaAM longform: {len(full_text)} символов\n")
            return full_text
            
        except Exception as e:
            flog.write(f"❌ GigaAM: {e}\n")
            import traceback
            flog.write(traceback.format_exc() + "\n")
            raise RuntimeError(str(e))       

def run_gigaam_with_diar(audio_path: str, hftoken: str, logfp: str, progress_callback=None) -> str:
    """GigaAM-v3 + pyannote 3.1 → SPEAKER_01 метки (с GPU оптимизацией)"""
    global device
    load_gigaam_models(hftoken)
    load_diarization_model(hftoken)

    with open(logfp, 'a', encoding='utf-8') as flog:
        flog.write("=== GIGAAM-V3 + PYANNOTE 3.1 (GPU) ===\n")
        flog.write(f"Device: {device}\n")
        
        try:
            # Загружаем аудио
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Перемещаем на GPU если доступно
            if device == "cuda":
                waveform = waveform.to(device)
                torch.cuda.reset_peak_memory_stats()  # Сброс статистики памяти
            
            # 1. Диаризация на GPU
            flog.write("🎤 Диаризация (GPU)...\n")
            
            # Проверяем, что модель на GPU
            if device == "cuda":
                # Явно указываем устройство при вызове пайплайна
                audio_dict = {'waveform': waveform, 'sample_rate': 16000}
                diar = DIAR_MODEL(audio_dict)
            else:
                audio_dict = {'waveform': waveform, 'sample_rate': 16000}
                diar = DIAR_MODEL(audio_dict)
            
            # Проверяем использование GPU
            if device == "cuda":
                gpu_mem = torch.cuda.max_memory_allocated() / 1024**2
                flog.write(f"   GPU память использовано: {gpu_mem:.1f} MB\n")
            
            # 2. Сегменты спикеров
            segments = []
            for turn, _, speaker in diar.itertracks(yield_label=True):
                if turn.duration > 0.8:  # Минимум 0.8 сек
                    segments.append((turn.start, turn.end, speaker))
            
            flog.write(f"Найдено сегментов: {len(segments)}\n")
            
            # 3. Транскрибация сегментов (GigaAM)
            full_transcript = []
            
            # Показываем прогресс для транскрибации
            total_segments = len(segments)
            flog.write(f"Начинаем транскрибацию {total_segments} сегментов...\n")
            
            for i, (start, end, speaker) in enumerate(segments):
                global STOP_FLAG
                if STOP_FLAG: break  # Проверка остановки
                
                # Вырезаем сегмент
                start_sample = int(start * 16000)
                end_sample = int(end * 16000)
                segment_wave = waveform[:, start_sample:end_sample]
                
                if segment_wave.shape[1] < 3200:  # Минимум 0.2 сек
                    continue
                
                # Перемещаем на CPU для сохранения во временный файл
                segment_wave_cpu = segment_wave.cpu() if device == "cuda" else segment_wave
                
                # Логируем прогресс каждые 10 сегментов или каждый 5%
                if i % 10 == 0 or (i * 100 // total_segments) > ((i-1) * 100 // total_segments):
                    progress_pct = (i * 100 // total_segments)
                    flog.write(f"Прогресс транскрибации: {i}/{total_segments} ({progress_pct}%)...\n")
                    if progress_callback:
                        progress_callback('progress', {'value': 25 + int(progress_pct * 0.25), 'label': f"Транскрибация... {progress_pct}% ({i}/{total_segments})"})
                
                # Временный WAV (GigaAM требует файл)
                # На Windows файл должен быть закрыт до torchaudio.save и unlink.
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    torchaudio.save(tmp_path, segment_wave_cpu, 16000)

                    # Транскрибация сегмента через transcribe_longform (для длинных сегментов)
                    try:
                        result = _transcribe_gigaam_longform(GIGA_MODEL, tmp_path)
                        # transcribe_longform возвращает список [{'transcription': 'text', 'boundaries': (start, end)}, ...]
                        transcript = ' '.join([r['transcription'] for r in result])
                    except Exception as e:
                        flog.write(f"⚠️ transcribe_longform для сегмента не сработал: {e}, пробуем transcribe...")
                        transcript = GIGA_MODEL.transcribe(tmp_path)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except FileNotFoundError:
                        pass
                
                speaker_id = speaker[-2:] if len(speaker) >= 2 else speaker[:2]
                full_transcript.append(f"SPEAKER_{speaker_id} [{start:.1f}-{end:.1f}s] {transcript.strip()}")
                flog.write(f"SPEAKER_{speaker_id}: {transcript}\n")
                
                # Показываем прогресс
                progress = int((i + 1) / len(segments) * 100)
                global UI_CALLBACK
                if UI_CALLBACK:
                    UI_CALLBACK('progress', {'value': 25 + int(progress * 0.25), 'label': f"Транскрибация сегментов... {progress}% ({i+1}/{len(segments)})"})
            
            result = "\n".join(full_transcript)
            if not result.strip() and not STOP_FLAG:
                # Fallback: вся аудио одним куском через transcribe_longform
                flog.write("Fallback: вся аудио одним куском через transcribe_longform\n")
                try:
                    long_result = _transcribe_gigaam_longform(GIGA_MODEL, audio_path)
                    result = ' '.join([r['transcription'] for r in long_result])
                except Exception as e:
                    flog.write(f"⚠️ transcribe_longform не сработал: {e}")
                    result = GIGA_MODEL.transcribe(audio_path)
            
            flog.write(f"\n=== ФИНАЛЬНЫЙ ТРАНСКРИПТ ===\n{result}\n")
            return result
            
        except Exception as e:
            flog.write(f"❌ Ошибка: {str(e)}\n")
            import traceback
            flog.write(traceback.format_exc() + "\n")
            
            # Не запускаем дорогой fallback после явной отмены пользователем.
            if STOP_FLAG:
                raise InterruptedError("Обработка отменена пользователем") from e

            # Критический fallback: только GigaAM через transcribe_longform
            try:
                flog.write("Критический fallback: пробуем transcribe_longform для всего файла...\n")
                long_result = _transcribe_gigaam_longform(GIGA_MODEL, audio_path)
                transcript = ' '.join([r['transcription'] for r in long_result])
                flog.write(f"✅ Fallback transcribe_longform: {len(transcript)} символов\n")
                return transcript
            except Exception as e2:
                error_msg = f"GigaAM полностью упал: {str(e2)}"
                flog.write(error_msg + "\n")
                flog.write(traceback.format_exc() + "\n")
                raise RuntimeError(error_msg) from e2

def gpu_info_text():
    try:
        lines = []
        lines.append(f"PyTorch: {torch.__version__}")
        lines.append(f"CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
            lines.append(f"CUDA: {torch.version.cuda}")
            try:
                cudnnv = torch.backends.cudnn.version()
                lines.append(f"cuDNN: {cudnnv}")
            except:
                lines.append("cuDNN: N/A")
            memalloc = torch.cuda.memory_allocated(0) / 1024**2
            memres = torch.cuda.memory_reserved(0) / 1024**2
            lines.append(f"GPU mem: allocated {memalloc:.1f} MiB, reserved {memres:.1f} MiB")
        return '\n'.join(lines)
    except Exception as e:
        return f"GPU info error: {e}"

def parse_transcript_from_stdout(stdout_text):
    raw_lines = []
    pretty_lines = []
    prefixes = ["Transcript", "Transcription", "Result"]
    for line in stdout_text.splitlines():
        raw_lines.append(line)
        s = line.strip()
        for pref in prefixes:
            if s.startswith(pref):
                pretty_lines.append(s[len(pref.strip()):].strip())
                break
    if not pretty_lines:
        pretty_lines = [stdout_text.strip()]
    return '\n'.join(raw_lines), '\n'.join(pretty_lines)

def run_ollama_summary(model_name, prompt_text, logfp):
    try:
        import requests
    except Exception:
        raise RuntimeError("requests not found. pip install requests")
    
    url = 'http://127.0.0.1:11434/api/generate'
    payload = {
        'model': model_name,
        'prompt': prompt_text,
        'stream': True
    }
    
    with open(logfp, 'a', encoding='utf-8') as flog:
        flog.write(f"=== OLLAMA SUMMARY: {model_name} ===\n")
        flog.write(f"Prompt length: {len(prompt_text)}\n")
    
    resp = requests.post(url, json=payload, stream=True, timeout=600)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text[:500]}")
    
    final_text = []
    for raw in resp.iter_lines():
        if STOP_FLAG:
            resp.close()
            raise InterruptedError("Суммаризация отменена пользователем")
        if not raw: continue
        try:
            obj = json.loads(raw.decode('utf-8', errors='ignore'))
            piece = obj.get('response', '')
            if piece: final_text.append(piece)
        except Exception:
            continue
    return ''.join(final_text)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(TITLE)
        self.geometry("900x700")
        config = load_config()
        self.files = config.get('files', [])
        self.last_result_dir = None
        global SELECTED_GIGAAM_MODEL
        SELECTED_GIGAAM_MODEL = config.get('gigaam_model', 'rnnt')
        self.ollama_models = []
        self.stop_flag = False
        self.progress_value = 0
        self.uiq = queue.Queue()
        
        # Top frame
        top = ttk.Frame(self, padding="10")
        top.pack(fill='x')
        btn_pick = ttk.Button(top, text="📁 Выбрать аудио", command=self.pick_files)
        btn_pick.pack(side='left')
        
        ttk.Label(top, text="GigaAM модель:").pack(side='left', padx=(12, 4))
        self.gigaam_model_options = {
            'RNNT v2 — рекомендуется': 'rnnt',
            'CTC v2 — быстрее': 'ctc',
            'RNNT v1': 'v1_rnnt',
            'CTC v1': 'v1_ctc',
        }
        selected_model = config.get('gigaam_model', 'rnnt')
        selected_label = next(
            (label for label, value in self.gigaam_model_options.items() if value == selected_model),
            'RNNT v2 — рекомендуется',
        )
        self.var_gigaam = tk.StringVar(value=selected_label)
        self.combo_gigaam = ttk.Combobox(
            top, textvariable=self.var_gigaam,
            values=list(self.gigaam_model_options), state='readonly', width=24,
        )
        self.combo_gigaam.pack(side='left', padx=(0, 4))
        self.combo_gigaam.bind('<<ComboboxSelected>>', lambda e: self.gigaam_model_changed())
        ttk.Button(top, text="📂 Папка модели", command=self.open_gigaam_model_folder).pack(side='left', padx=(0, 10))
        
        ttk.Label(top, text="HF Token (GigaAM + pyannote):").pack(side='left')
        # Загружаем сохраненный токен
        self.var_hf = tk.StringVar(value=config['hf_token'])
        hf_entry = ttk.Entry(top, textvariable=self.var_hf, width=32, show="*")
        hf_entry.pack(side='left', padx=(4, 10))
        hf_entry.bind('<Control-v>', lambda e: hf_entry.event_generate('<<Paste>>'))
        # Сохраняем токен при изменении (с задержкой, чтобы не спамить)
        self._save_timer = None
        def delayed_save(*args):
            if self._save_timer:
                self.after_cancel(self._save_timer)
            self._save_timer = self.after(1000, self.save_settings)  # Сохраняем через 1 сек после последнего изменения
        hf_entry.bind('<KeyRelease>', delayed_save)
        hf_entry.bind('<FocusOut>', lambda e: self.save_settings())
        
        # Middle frame (Ollama + Diarization)
        mid = ttk.Frame(self, padding="10")
        mid.pack(fill='x')
        
        # Checkbox for speaker diarization
        self.var_diarize = tk.BooleanVar(value=config.get('diarize', False))
        self.chk_diarize = ttk.Checkbutton(mid, text="🎤 Диаризация спикеров", variable=self.var_diarize)
        self.chk_diarize.pack(side='left')
        
        # Export format checkboxes
        ttk.Label(mid, text="Экспорт:").pack(side='left', padx=(12, 4))
        self.var_export_srt = tk.BooleanVar(value=config.get('export_srt', True))
        ttk.Checkbutton(mid, text="SRT", variable=self.var_export_srt).pack(side='left')
        self.var_export_vtt = tk.BooleanVar(value=config.get('export_vtt', False))
        ttk.Checkbutton(mid, text="VTT", variable=self.var_export_vtt).pack(side='left')
        self.var_export_json = tk.BooleanVar(value=config.get('export_json', False))
        ttk.Checkbutton(mid, text="JSON", variable=self.var_export_json).pack(side='left')
        self.var_export_docx = tk.BooleanVar(value=config.get('export_docx', False))
        ttk.Checkbutton(mid, text="DOCX", variable=self.var_export_docx).pack(side='left')
        
        ttk.Separator(mid, orient='vertical').pack(side='left', fill='y', padx=10, pady=2)
        
        self.var_summarize = tk.BooleanVar(value=config.get('summarize', True))
        ttk.Checkbutton(mid, text="Сводка Ollama", variable=self.var_summarize).pack(side='left')
        self.var_ollama = tk.StringVar(value=config['ollama_model'])
        self.combo_ollama = ttk.Combobox(mid, textvariable=self.var_ollama, state="readonly", values=[], width=32)
        self.combo_ollama.pack(side='left', padx=(6, 6))
        # Сохраняем модель при изменении
        self.combo_ollama.bind('<<ComboboxSelected>>', lambda e: self.save_settings())
        ttk.Button(mid, text="🔄 Обновить", command=self.refresh_ollama_models).pack(side='left')

        # Список выбранных файлов сохраняется между запусками приложения.
        files_frame = ttk.Frame(self, padding="10")
        files_frame.pack(fill='x')
        ttk.Label(files_frame, text="Файлы для обработки:").pack(anchor='w')
        files_row = ttk.Frame(files_frame)
        files_row.pack(fill='x')
        self.files_listbox = tk.Listbox(files_row, height=4, exportselection=False)
        self.files_listbox.pack(side='left', fill='x', expand=True)
        ttk.Button(files_row, text="🗑 Очистить", command=self.clear_files).pack(side='left', padx=(8, 0), anchor='n')
        self.refresh_files_list()

        # Сохраняем флажки сразу после изменения.
        for variable in (self.var_diarize, self.var_summarize, self.var_export_srt,
                         self.var_export_vtt, self.var_export_json, self.var_export_docx):
            variable.trace_add('write', lambda *_: self.save_settings())
        
        # Progress frame
        pr = ttk.Frame(self, padding="10")
        pr.pack(fill='x')
        self.var_progress = tk.IntVar(value=0)
        self.progress = ttk.Progressbar(pr, maximum=100, variable=self.var_progress)
        self.progress.pack(fill='x')
        self.var_stage = tk.StringVar(value="Готов к запуску")
        ttk.Label(pr, textvariable=self.var_stage).pack(anchor='w')
        
        # Output frame
        out = ttk.Frame(self, padding="10")
        out.pack(fill='both', expand=True)
        self.txt = tk.Text(out, wrap='word', height=20, font=('Consolas', 9))
        self.txt.pack(fill='both', expand=True)
        self.txt.insert('end', "🚀 GigaAM-v3 + pyannote + Ollama\n\n")
        
        # Bottom frame
        bottom = ttk.Frame(self, padding="10")
        bottom.pack(fill='x')
        self.btn_start = ttk.Button(bottom, text="▶️ СТАРТ", command=self.start_thread)
        self.btn_start.pack(side='left')
        ttk.Button(bottom, text="⏹️ СТОП", command=self.set_stop).pack(side='left', padx=(6, 0))
        self.btn_open_results = ttk.Button(
            bottom,
            text="📂 Открыть результаты",
            command=self.open_results_folder,
            state='disabled',
        )
        self.btn_open_results.pack(side='left', padx=(6, 0))
        
        # Auto refresh
        self.after(200, self.refresh_ollama_models)
        self.after(33, self.process_ui_queue)
        
        # Сохраняем настройки при закрытии окна
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def ui_post(self, kind, payload):
        # Фоновая транскрибация может поставить несколько событий в очередь
        # до следующего такта Tk. Сохраняем значение сразу, чтобы следующий
        # этап не вычислялся от устаревшего прогресса.
        if kind == 'progress':
            self.progress_value = int(payload.get('value', 0))
        self.uiq.put((kind, payload))
    
    def process_ui_queue(self):
        try:
            while True:
                kind, data = self.uiq.get_nowait()
                if kind == 'log':
                    self.txt.insert('end', data.get('text', ''))
                    self.txt.see('end')
                elif kind == 'progress':
                    progress_value = int(data.get('value', 0))
                    self.progress_value = progress_value
                    self.var_progress.set(progress_value)
                    self.var_stage.set(data.get('label', ''))
                elif kind == 'stage_text':
                    self.var_stage.set(data.get('label', ''))
                elif kind == 'start_btn_state':
                    self.btn_start.configure(state=data.get('state', 'normal'))
                elif kind == 'open_results_state':
                    self.btn_open_results.configure(state=data.get('state', 'disabled'))
                elif kind == 'set_ollama_values':
                    vals = data.get('values', [])
                    self.combo_ollama['values'] = vals
                    self.combo_ollama.configure(state='readonly' if vals else 'disabled')
                elif kind == 'gigaam_model_state':
                    self.combo_gigaam.configure(state=data.get('state', 'readonly'))
        except queue.Empty:
            pass
        finally:
            self.after(33, self.process_ui_queue)
    
    def log_gui(self, text):
        self.ui_post('log', {'text': text + '\n'})
    
    def set_stage(self, title, add_percent):
        self.progress_value = min(100, getattr(self, 'progress_value', 0) + add_percent)
        self.ui_post('progress', {'value': self.progress_value, 'label': f"{title} {self.progress_value}%"})
    
    def save_settings(self):
        """Сохраняет текущие настройки в файл"""
        try:
            hf_token = self.var_hf.get().strip()
            ollama_model = self.var_ollama.get() or 'mistral:7b'
            save_config(
                hf_token,
                ollama_model,
                self.files,
                self.var_diarize.get(),
                self.var_export_srt.get(),
                self.var_export_vtt.get(),
                self.var_export_json.get(),
                self.var_export_docx.get(),
                self.gigaam_model_options.get(self.var_gigaam.get(), 'rnnt'),
                self.var_summarize.get(),
            )
        except Exception as e:
            print(f"❌ Ошибка сохранения настроек: {e}")
            import traceback
            traceback.print_exc()
    
    def on_closing(self):
        """Обработчик закрытия окна - сохраняет настройки перед выходом"""
        self.save_settings()
        self.destroy()

    def gigaam_model_changed(self):
        """Применяет выбранную модель к следующему запуску транскрибации."""
        global SELECTED_GIGAAM_MODEL, GIGA_MODEL
        SELECTED_GIGAAM_MODEL = self.gigaam_model_options.get(self.var_gigaam.get(), 'rnnt')
        GIGA_MODEL = None
        self.save_settings()
        self.log_gui(f"🧠 Выбрана модель GigaAM: {self.var_gigaam.get()}")

    def open_gigaam_model_folder(self):
        """Открывает папку, в которой GigaAM хранит checkpoint-модели."""
        model_dir = os.path.expanduser('~/.cache/gigaam')
        os.makedirs(model_dir, exist_ok=True)
        try:
            if os.name == 'nt':
                os.startfile(model_dir)
            else:
                import subprocess
                subprocess.Popen(['xdg-open', model_dir])
        except Exception as e:
            messagebox.showerror("Модели GigaAM", f"Не удалось открыть папку:\n{e}")
    
    def pick_files(self):
        files = filedialog.askopenfilenames(
            title="Выбери аудио",
            filetypes=[
                ("Audio", "*.mp3 *.wav *.m4a *.flac *.ogg *.aac"),
                ("All", "*.*")
            ]
        )
        if files:
            existing = set(self.files)
            self.files.extend(path for path in files if path not in existing)
            self.refresh_files_list()
            self.save_settings()
            self.log_gui(f"📁 Выбрано файлов: {len(self.files)}")

    def refresh_files_list(self):
        """Обновляет видимый список файлов и убирает исчезнувшие пути."""
        self.files = [path for path in self.files if os.path.isfile(path)]
        if not hasattr(self, 'files_listbox'):
            return
        self.files_listbox.delete(0, tk.END)
        for path in self.files:
            self.files_listbox.insert(tk.END, path)

    def clear_files(self):
        """Очищает список выбранных файлов и сохраняет это состояние."""
        self.files = []
        self.refresh_files_list()
        self.save_settings()
        self.log_gui("🗑 Список файлов очищен")

    def open_results_folder(self):
        """Открывает папку результата выбранной записи или последнего файла."""
        result_dir = self.last_result_dir
        if hasattr(self, 'files_listbox'):
            selected = self.files_listbox.curselection()
            if selected:
                result_dir = self.find_latest_result_folder(self.files[selected[0]])

        if not result_dir or not os.path.isdir(result_dir):
            messagebox.showinfo("Результаты", "Папка результатов ещё не создана.")
            return
        try:
            if os.name == 'nt':
                os.startfile(result_dir)
            else:
                import subprocess
                subprocess.Popen(['xdg-open', result_dir])
        except Exception as e:
            messagebox.showerror("Результаты", f"Не удалось открыть папку:\n{e}")

    def find_latest_result_folder(self, audio_path):
        """Находит последнюю папку результатов для конкретной записи."""
        root = parent_dir(audio_path)
        prefix = f"gigaam_{safe_result_name(audio_path)}_"
        folders = [
            entry for entry in os.scandir(root)
            if entry.is_dir() and entry.name.startswith(prefix)
        ]
        if not folders:
            return None
        return max(folders, key=lambda entry: entry.stat().st_mtime).path
    
    def refresh_ollama_models(self):
        self.combo_ollama.configure(state='disabled')
        self.combo_ollama.set('')
        self.combo_ollama['values'] = []
        self.log_gui("🔄 Обновление Ollama...")
        threading.Thread(target=self.refresh_ollama_models_worker, daemon=True).start()
    
    def refresh_ollama_models_worker(self):
        try:
            models = detect_ollama_models()
            if not models:
                self.ui_post('log', {'text': '⚠️ Ollama не найдена. Запусти: ollama serve'})
                self.ui_post('set_ollama_values', {'values': []})
            else:
                self.ollama_models = models
                self.ui_post('set_ollama_values', {'values': models})
                self.ui_post('log', {'text': f"✅ Ollama модели ({len(models)}): {', '.join(models[:3])}..."})
        except Exception as e:
            self.ui_post('log', {'text': f"❌ Ollama error: {e}"})
    
    def set_stop(self):
        global STOP_FLAG
        STOP_FLAG = True
        self.stop_flag = True
        self.log_gui("⏹️ Остановка...")
    
    def start_thread(self):
        if not self.files:
            messagebox.showwarning("Ошибка", "Выбери аудио файлы!")
            return
        if self.var_summarize.get() and not self.var_ollama.get():
            messagebox.showwarning("Ollama", "Выбери Ollama модель!")
            return
        hf_token = self.var_hf.get().strip()
        if not hf_token:
            response = messagebox.askyesno(
                "HF Token отсутствует",
                "HuggingFace токен не введен. GigaAM требует токен для работы.\n\n"
                "Получить токен: https://huggingface.co/settings/tokens\n\n"
                "Продолжить без токена? (может не работать)"
            )
            if not response:
                return
        
        global STOP_FLAG, UI_CALLBACK
        STOP_FLAG = False
        UI_CALLBACK = self.ui_post  # Устанавливаем callback для обновления UI
        self.stop_flag = False
        self.btn_start.configure(state='disabled')
        # Смена модели во время работы сбрасывала глобальную модель, которую
        # использует фоновый поток. Применяем новый выбор после завершения.
        self.combo_gigaam.configure(state='disabled')
        files = self.files[:]
        threading.Thread(target=self.run_pipeline, args=(files,), daemon=True).start()
    
    def run_pipeline(self, files):
        try:
            self.progress_value = 0
            self.ui_post('progress', {'value': 0, 'label': ''})
            succeeded = []
            failed = []
            self.log_gui("\n🚀 === ГИГАЧАД ТРАНСКРИБАЦИЯ СТАРТ ===")
            
            # GPU info
            gpu_txt = gpu_info_text()
            self.log_gui(gpu_txt)
            
            global STOP_FLAG
            for idx, ap in enumerate(files, 1):
                if STOP_FLAG: break
                self.log_gui(f"\n📄 [{idx}/{len(files)}] {os.path.basename(ap)}")
                
                outdir = make_result_folder(ap)
                self.last_result_dir = outdir
                self.ui_post('open_results_state', {'state': 'normal'})
                result_name = safe_result_name(ap)
                logfp = os.path.join(outdir, f'{result_name}_gigaam_log.txt')
                rawfp = os.path.join(outdir, f'{result_name}_transcript_raw.txt')
                cleanfp = os.path.join(outdir, f'{result_name}_transcript.txt')
                sumfp = os.path.join(outdir, f'{result_name}_summary.txt')
                
                with open(logfp, 'a', encoding='utf-8') as lf:
                    lf.write(f"{datetime.datetime.now()} START\n")
                    lf.write(gpu_txt + "\n")
                
                # Этап 1: GigaAM + диаризация (25-75%)
                if STOP_FLAG: break
                self.set_stage(STAGES_RU[0], 25)
                try:
                    hf_token = self.var_hf.get().strip()
                    
                    # Выбор метода транскрипции
                    if self.var_diarize.get():
                        self.log_gui("🎤 Используем диаризацию спикеров...")
                        stdout_text = run_gigaam_with_diar(ap, hf_token, logfp, self.ui_post)
                    else:
                        stdout_text = run_gigaam_simple(ap, logfp, hf_token)
                    
                    if STOP_FLAG:
                        break

                    raw_out, pretty = parse_transcript_from_stdout(stdout_text)
                    
                    with open(rawfp, 'w', encoding='utf-8') as f:
                        f.write(raw_out)
                    with open(cleanfp, 'w', encoding='utf-8') as f:
                        f.write(pretty)
                    
                    # Экспорт в выбранные форматы
                    if self.var_export_srt.get():
                        srt_path = os.path.join(outdir, f'{result_name}_transcript.srt')
                        if export_srt(pretty, srt_path):
                            self.log_gui(f"✅ SRT сохранён: {os.path.basename(srt_path)}")
                    if self.var_export_vtt.get():
                        vtt_path = os.path.join(outdir, f'{result_name}_transcript.vtt')
                        if export_vtt(pretty, vtt_path):
                            self.log_gui(f"✅ VTT сохранён: {os.path.basename(vtt_path)}")
                    if self.var_export_json.get():
                        json_path = os.path.join(outdir, f'{result_name}_transcript.json')
                        if export_json(pretty, json_path):
                            self.log_gui(f"✅ JSON сохранён: {os.path.basename(json_path)}")
                    
                    self.set_stage(STAGES_RU[2], 25)
                except Exception as e:
                    error_msg = str(e)
                    self.log_gui(f"❌ GigaAM ошибка: {error_msg}")
                    # Если это ошибка токена, показываем более понятное сообщение
                    if "HF_TOKEN" in error_msg or "token" in error_msg.lower():
                        self.log_gui("\n💡 Решение:")
                        self.log_gui("   1. Получите HuggingFace токен: https://huggingface.co/settings/tokens")
                        self.log_gui("   2. Введите токен в поле 'HF Token' в интерфейсе")
                        self.log_gui("   3. Запустите транскрипцию снова\n")
                    with open(logfp, 'a', encoding='utf-8') as lf:
                        lf.write(f"ERROR GIGAAM: {error_msg}\n")
                    failed.append(os.path.basename(ap))
                    self.set_stage("GigaAM ошибка", 10)
                    continue
                
                summary = ""
                # Суммаризация — необязательный этап: транскрипция должна
                # работать и на компьютере без Ollama.
                if self.var_summarize.get():
                    if STOP_FLAG: break
                    self.set_stage(STAGES_RU[3], 25)
                    prompt = PROMPTTEMPLATE.replace("{TRANSCRIPT}", pretty)

                    try:
                        summary = run_ollama_summary(self.var_ollama.get(), prompt, logfp)
                        with open(sumfp, 'w', encoding='utf-8') as f:
                            f.write(summary)
                        self.log_gui(f"✅ Суммаризация: {len(summary)} символов")

                    except Exception as e:
                        if STOP_FLAG:
                            break
                        self.log_gui(f"⚠️ Ollama ошибка: {e}")
                        with open(logfp, 'a', encoding='utf-8') as lf:
                            lf.write(f"ERROR OLLAMA: {e}\n")

                # DOCX содержит сводку, если она была получена, но сам
                # транскрипт экспортируется даже при недоступном Ollama.
                if self.var_export_docx.get():
                    docx_path = os.path.join(outdir, f'{result_name}_transcript.docx')
                    if export_docx(pretty, docx_path, summary):
                        self.log_gui(f"✅ DOCX сохранён: {os.path.basename(docx_path)}")
                
                self.set_stage(STAGES_RU[4], 0)
                self.log_gui(f"📁 Результаты в: {outdir}")
                self.log_gui(f"   📄 {os.path.basename(rawfp)} (с метками спикеров)")
                self.log_gui(f"   📄 {os.path.basename(cleanfp)} (чистый текст)")
                self.log_gui(f"   📄 {os.path.basename(sumfp)} (сводка Ollama)")
                self.log_gui(f"   📋 {os.path.basename(logfp)} (лог)")
                succeeded.append(os.path.basename(ap))
            
            if not self.stop_flag:
                self.ui_post('progress', {
                    'value': 100,
                    'label': f"Готово: успешно {len(succeeded)}, с ошибками {len(failed)}",
                })
                if failed:
                    self.log_gui(f"⚠️ Не обработаны файлы: {', '.join(failed)}")
            else:
                self.ui_post('stage_text', {'label': 'Остановлено'})
                
        except Exception as e:
            self.log_gui(f"❌ Критическая ошибка: {e}")
        finally:
            self.ui_post('start_btn_state', {'state': 'normal'})
            self.ui_post('gigaam_model_state', {'state': 'readonly'})

if __name__ == '__main__':
    try:
        if os.name == 'nt':
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    
    # Выводим информацию о GPU при запуске
    print("\n" + "="*50)
    print("ИНФОРМАЦИЯ О СИСТЕМЕ:")
    print("="*50)
    print(gpu_info_text())
    print("="*50 + "\n")
    
    app = App()
    app.mainloop()
