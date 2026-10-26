"""
Минимальные тесты парсинга таймкодов транскрипта (main.parse_transcript_timestamps).

Форматы, которые реально генерирует приложение:
  * run_gigaam_simple:      "[12.3-45.6s] текст"
  * run_gigaam_with_diar:   "SPEAKER_01 [12.3-45.6s] текст"
Формат без суффикса "s" также должен поддерживаться.
"""

import sys
from pathlib import Path

# Позволяем импортировать main.py из корня проекта
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import parse_transcript_timestamps  # noqa: E402


def test_simple_format_with_s_suffix():
    segments = parse_transcript_timestamps("[12.3-45.6s] Привет мир")
    assert segments == [(12.3, 45.6, "Привет мир")]


def test_format_without_s_suffix_is_supported():
    segments = parse_transcript_timestamps("[12.3-45.6] Привет мир")
    assert segments == [(12.3, 45.6, "Привет мир")]


def test_multiple_consecutive_segments():
    text = (
        "[0.0-3.2s] Первый\n"
        "[3.5-10.1s] Второй\n"
        "[11.0-20.4s] Третий"
    )
    segments = parse_transcript_timestamps(text)
    assert segments == [
        (0.0, 3.2, "Первый"),
        (3.5, 10.1, "Второй"),
        (11.0, 20.4, "Третий"),
    ]


def test_speaker_segment():
    text = "SPEAKER_01 [0.0-3.2s] Привет, как дела?"
    segments = parse_transcript_timestamps(text)
    assert segments == [(0.0, 3.2, "SPEAKER_01 Привет, как дела?")]


def test_integer_timestamps():
    segments = parse_transcript_timestamps("[10-20s] Текст")
    assert segments == [(10.0, 20.0, "Текст")]