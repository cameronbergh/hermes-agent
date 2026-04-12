"""Tests for tools/tts_tool.py platform-specific output format behavior."""

import json


def test_signal_keeps_mlx_tts_as_mp3(monkeypatch, tmp_path):
    from tools import tts_tool

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "signal")
    monkeypatch.setattr(tts_tool, "DEFAULT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "mlx"})
    monkeypatch.setattr(tts_tool, "_get_provider", lambda cfg: "mlx")
    monkeypatch.setattr(tts_tool, "check_tts_requirements", lambda: True)

    def fake_generate_mlx(text, output_path, tts_config):
        with open(output_path, "wb") as f:
            f.write(b"ID3fake-mp3")
        return output_path

    def fail_convert(_path):
        raise AssertionError("Signal/non-Telegram TTS should not force OGG conversion")

    monkeypatch.setattr(tts_tool, "_generate_mlx", fake_generate_mlx)
    monkeypatch.setattr(tts_tool, "_convert_to_opus", fail_convert)

    result = json.loads(tts_tool.text_to_speech_tool("hello signal"))

    assert result["success"] is True
    assert result["file_path"].endswith(".mp3")
    assert result["voice_compatible"] is False
    assert result["media_tag"].startswith("MEDIA:")
    assert "[[audio_as_voice]]" not in result["media_tag"]


def test_telegram_converts_mlx_tts_to_ogg(monkeypatch, tmp_path):
    from tools import tts_tool

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setattr(tts_tool, "DEFAULT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "mlx"})
    monkeypatch.setattr(tts_tool, "_get_provider", lambda cfg: "mlx")
    monkeypatch.setattr(tts_tool, "check_tts_requirements", lambda: True)

    def fake_generate_mlx(text, output_path, tts_config):
        with open(output_path, "wb") as f:
            f.write(b"ID3fake-mp3")
        return output_path

    def fake_convert(path):
        ogg_path = path.rsplit(".", 1)[0] + ".ogg"
        with open(ogg_path, "wb") as f:
            f.write(b"OggSfake-opus")
        return ogg_path

    monkeypatch.setattr(tts_tool, "_generate_mlx", fake_generate_mlx)
    monkeypatch.setattr(tts_tool, "_convert_to_opus", fake_convert)

    result = json.loads(tts_tool.text_to_speech_tool("hello telegram"))

    assert result["success"] is True
    assert result["file_path"].endswith(".ogg")
    assert result["voice_compatible"] is True
    assert result["media_tag"].startswith("[[audio_as_voice]]\nMEDIA:")
