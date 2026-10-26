"""
Минимальный тест fallback-логики run_gigaam_with_diar без загрузки моделей/GPU.

Имитирует ситуацию, в которой диаризация падает, а критический longform
fallback также завершается ошибкой. Проверяем, что одинаковый вызов
transcribe_longform() не выполняется повторно.
"""

import sys
import tempfile
import os
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402


class FakeGigaModel:
    """Фейк-модель, у которой longform также падает."""

    def __init__(self):
        self.transcribe_calls = 0
        self.longform_calls = 0

    def transcribe(self, audio_path):
        self.transcribe_calls += 1
        raise ValueError("Too long wav file, use 'transcribe_longform' method.")

    def transcribe_longform(self, audio_path):
        self.longform_calls += 1
        raise RuntimeError("longform-вызов упал")


def test_diar_last_fallback_does_not_repeat_the_same_longform_call():
    fake_model = FakeGigaModel()
    old_device = main.device
    main.device = "cpu"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            logfp = str(Path(tmpdir) / "gigaam_log.txt")
            with mock.patch.object(main, "GIGA_MODEL", fake_model), \
                 mock.patch.object(main, "DIAR_MODEL", object()), \
                 mock.patch.object(main, "load_gigaam_models", return_value=None), \
                 mock.patch.object(main, "load_diarization_model", return_value=None), \
                 mock.patch.object(main.torchaudio, "load",
                                   side_effect=RuntimeError("audio load failed")):
                try:
                    main.run_gigaam_with_diar("fake.wav", "token", logfp)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("Ожидалась ошибка единственного fallback-вызова")
    finally:
        main.device = old_device

    # Повтор того же вызова не выполняется.
    assert fake_model.longform_calls == 1
    # Обычный transcribe больше не должен вызываться
    assert fake_model.transcribe_calls == 0


def test_gigaam_loader_can_use_module_level_sys_before_loading_model():
    """Регрессия: локальный ``import sys`` не должен затенять sys выше."""
    previous_model = main.GIGA_MODEL
    previous_name = main.GIGAAM_MODEL_NAME
    previous_selected = getattr(main, 'SELECTED_GIGAAM_MODEL', None)
    fake_loaded_model = object()
    main.GIGA_MODEL = None
    main.GIGAAM_MODEL_NAME = 'rnnt'
    main.SELECTED_GIGAAM_MODEL = 'rnnt'
    try:
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(main, '_load_gigaam_model', return_value=fake_loaded_model):
            main.load_gigaam_models('test-token')
        assert main.GIGA_MODEL is fake_loaded_model
    finally:
        main.GIGA_MODEL = previous_model
        main.GIGAAM_MODEL_NAME = previous_name
        if previous_selected is None:
            delattr(main, 'SELECTED_GIGAAM_MODEL')
        else:
            main.SELECTED_GIGAAM_MODEL = previous_selected
