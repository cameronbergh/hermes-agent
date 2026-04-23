"""Tests for the Hermes Mumble bridge tool wrappers."""

import json
import struct
import wave

from tools import mumble_tool


class TestMumbleConfig:
    def test_default_bridge_url_points_at_local_bridge(self, monkeypatch):
        monkeypatch.delenv("MUMBLE_BRIDGE_URL", raising=False)
        assert mumble_tool._get_bridge_base_url() == "http://127.0.0.1:8787"

    def test_bridge_url_env_override_is_respected(self, monkeypatch):
        monkeypatch.setenv("MUMBLE_BRIDGE_URL", "http://localhost:9999/")
        assert mumble_tool._get_bridge_base_url() == "http://localhost:9999"


class TestMumbleStatusAndConnect:
    def test_status_returns_bridge_payload(self, monkeypatch):
        monkeypatch.setattr(
            mumble_tool,
            "_request_json",
            lambda method, path, payload=None, files=None, form=None: {"ok": True, "status": {"connected": False}},
        )
        result = json.loads(mumble_tool.mumble_status({}))
        assert result["ok"] is True
        assert result["status"]["connected"] is False

    def test_connect_uses_env_defaults_when_args_missing(self, monkeypatch):
        monkeypatch.setenv("MUMBLE_SERVER", "10.42.0.1")
        monkeypatch.setenv("MUMBLE_PORT", "64738")
        monkeypatch.setenv("MUMBLE_USERNAME", "Hermes")
        monkeypatch.setenv("MUMBLE_PASSWORD", "")
        monkeypatch.setenv("MUMBLE_INSECURE", "true")

        seen = {}

        def fake_request(method, path, payload=None, files=None, form=None):
            seen["method"] = method
            seen["path"] = path
            seen["payload"] = payload
            return {"connected": True, "ready": True}

        monkeypatch.setattr(mumble_tool, "_request_json", fake_request)
        result = json.loads(mumble_tool.mumble_connect({}))
        assert result["connected"] is True
        assert seen["method"] == "POST"
        assert seen["path"] == "/connect"
        assert seen["payload"]["server"] == "10.42.0.1"
        assert seen["payload"]["port"] == 64738
        assert seen["payload"]["username"] == "Hermes"
        assert seen["payload"]["insecure"] is True


class TestMumbleMessaging:
    def test_message_posts_text_payload(self, monkeypatch):
        seen = {}

        def fake_request(method, path, payload=None, files=None, form=None):
            seen["method"] = method
            seen["path"] = path
            seen["payload"] = payload
            return {"ok": True}

        monkeypatch.setattr(mumble_tool, "_request_json", fake_request)
        result = json.loads(mumble_tool.mumble_message({"message": "hello", "target": "channel"}))
        assert result["ok"] is True
        assert seen["path"] == "/message"
        assert seen["payload"] == {"message": "hello", "target": "channel", "username": None}

    def test_join_posts_form_payload(self, monkeypatch):
        seen = {}

        def fake_request(method, path, payload=None, files=None, form=None):
            seen["method"] = method
            seen["path"] = path
            seen["form"] = form
            return {"ok": True}

        monkeypatch.setattr(mumble_tool, "_request_json", fake_request)
        result = json.loads(mumble_tool.mumble_join({"channel": "Root/Stage"}))
        assert result["ok"] is True
        assert seen["method"] == "POST"
        assert seen["path"] == "/join"
        assert seen["form"] == {"channel": "Root/Stage"}

    def test_audio_file_upload_reads_local_file(self, tmp_path, monkeypatch):
        wav_path = tmp_path / "clip.wav"
        wav_path.write_bytes(b"RIFF....WAVEfmt ")

        seen = {}

        def fake_request(method, path, payload=None, files=None, form=None):
            seen["method"] = method
            seen["path"] = path
            seen["files"] = files
            return {"ok": True, "bytes": 16}

        monkeypatch.setattr(mumble_tool, "_request_json", fake_request)
        result = json.loads(mumble_tool.mumble_send_audio_file({"path": str(wav_path)}))
        assert result["ok"] is True
        assert result["bytes"] == 16
        assert seen["method"] == "POST"
        assert seen["path"] == "/audio/wav"
        assert seen["files"]["file"][0] == "clip.wav"
        assert seen["files"]["file"][1] == wav_path.read_bytes()

    def test_transcribe_audio_file_posts_form_and_file(self, tmp_path, monkeypatch):
        wav_path = tmp_path / "clip.wav"
        wav_path.write_bytes(b"RIFF....WAVEfmt ")

        seen = {}

        def fake_request(method, path, payload=None, files=None, form=None):
            seen["method"] = method
            seen["path"] = path
            seen["files"] = files
            seen["form"] = form
            return {"ok": True, "transcript": "hello world"}

        monkeypatch.setattr(mumble_tool, "_request_json", fake_request)
        result = json.loads(mumble_tool.mumble_transcribe_audio_file({"path": str(wav_path), "model": "medium", "language": "en", "speaker": "Cam"}))
        assert result["ok"] is True
        assert result["transcript"] == "hello world"
        assert seen["method"] == "POST"
        assert seen["path"] == "/transcribe/wav"
        assert seen["form"] == {"model": "medium", "language": "en", "speaker": "Cam"}
        assert seen["files"]["file"][0] == "clip.wav"

    def test_transcripts_fetches_transcript_events(self, monkeypatch):
        seen = {}

        def fake_request(method, path, payload=None, files=None, form=None):
            seen["path"] = path
            return {"events": [{"kind": "transcript_received", "data": {"transcript": "hi"}}]}

        monkeypatch.setattr(mumble_tool, "_request_json", fake_request)
        result = json.loads(mumble_tool.mumble_transcripts({"after": 2}))
        assert result["events"][0]["kind"] == "transcript_received"
        assert seen["path"] == "/transcripts?after=2"

    def test_disconnect_posts_without_payload(self, monkeypatch):
        seen = {}

        def fake_request(method, path, payload=None, files=None, form=None):
            seen["method"] = method
            seen["path"] = path
            seen["payload"] = payload
            return {"ok": True}

        monkeypatch.setattr(mumble_tool, "_request_json", fake_request)
        result = json.loads(mumble_tool.mumble_disconnect({}))
        assert result["ok"] is True
        assert seen["method"] == "POST"
        assert seen["path"] == "/disconnect"
        assert seen["payload"] is None


class TestMumbleSpeak:
    def test_speak_generates_tts_and_uploads_bridge_ready_wav(self, tmp_path, monkeypatch):
        wav_path = tmp_path / "tts.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(struct.pack("<480h", *([0] * 480)))

        monkeypatch.setattr(mumble_tool, "_current_bridge_status", lambda: {"connected": True, "ready": True, "server": "10.42.0.1"})

        seen = {}

        def fake_request(method, path, payload=None, files=None, form=None):
            seen["method"] = method
            seen["path"] = path
            seen["files"] = files
            return {"ok": True, "bytes": 960}

        monkeypatch.setattr(mumble_tool, "_request_json", fake_request)

        def fake_tts(text, output_path=None):
            assert text == "hello mumble"
            return json.dumps({"success": True, "file_path": str(wav_path)})

        monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", fake_tts)

        result = json.loads(mumble_tool.mumble_speak({"text": "hello mumble"}))
        assert result["ok"] is True
        assert result["bytes"] == 960
        assert seen["method"] == "POST"
        assert seen["path"] == "/audio/wav"
        assert seen["files"]["file"][0].endswith(".wav")

    def test_speak_requires_connected_ready_bridge(self, monkeypatch):
        monkeypatch.setattr(mumble_tool, "_current_bridge_status", lambda: {"connected": False, "ready": False})
        result = json.loads(mumble_tool.mumble_speak({"text": "hi"}))
        assert "not connected" in result["error"]
