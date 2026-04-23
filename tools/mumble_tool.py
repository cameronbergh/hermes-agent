"""Hermes Mumble bridge tool wrappers.

This module exposes the local FastAPI Mumble bridge as LLM-callable tools.
The bridge itself runs on the Mac at http://127.0.0.1:8787 by default.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from tools.registry import registry

_BRIDGE_URL = ""
_BRIDGE_TIMEOUT = 15


def _get_bridge_base_url() -> str:
    """Return the base URL for the local Mumble bridge."""
    return (_BRIDGE_URL or os.getenv("MUMBLE_BRIDGE_URL", "http://127.0.0.1:8787")).rstrip("/")


def _get_env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _request_json(
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    files: Optional[dict[str, Any]] = None,
    form: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Issue a request against the bridge and return decoded JSON."""
    url = f"{_get_bridge_base_url()}{path}"
    try:
        response = requests.request(
            method,
            url,
            json=payload if files is None and form is None else None,
            data=form,
            files=files,
            timeout=_BRIDGE_TIMEOUT,
        )
        response.raise_for_status()
        if not response.content:
            return {"ok": True}
        try:
            data = response.json()
        except Exception:
            return {"ok": True, "text": response.text}
        if isinstance(data, dict):
            return data
        return {"ok": True, "data": data}
    except Exception as exc:
        return {"error": f"Mumble bridge request failed: {type(exc).__name__}: {exc}"}


def _current_bridge_status() -> Dict[str, Any]:
    status = _request_json("GET", "/status")
    if isinstance(status, dict):
        return status
    return {}


def _default_connect_payload(args: Dict[str, Any]) -> Dict[str, Any]:
    status = _current_bridge_status()
    return {
        "server": args.get("server") or os.getenv("MUMBLE_SERVER") or status.get("server"),
        "port": int(args.get("port") or os.getenv("MUMBLE_PORT") or status.get("port") or 64738),
        "username": args.get("username") or os.getenv("MUMBLE_USERNAME") or status.get("username") or "Hermes",
        "password": args.get("password") if args.get("password") is not None else os.getenv("MUMBLE_PASSWORD", ""),
        "insecure": bool(args.get("insecure") if args.get("insecure") is not None else _get_env_bool("MUMBLE_INSECURE", True)),
        "channel": args.get("channel") or os.getenv("MUMBLE_CHANNEL") or None,
        "tokens": args.get("tokens") or [],
        "certfile": args.get("certfile") or None,
        "keyfile": args.get("keyfile") or None,
        "stereo": bool(args.get("stereo") if args.get("stereo") is not None else _get_env_bool("MUMBLE_STEREO", False)),
        "receive_sound": bool(args.get("receive_sound") if args.get("receive_sound") is not None else _get_env_bool("MUMBLE_RECEIVE_SOUND", True)),
    }


def mumble_status(args: Dict[str, Any]) -> str:
    """Return the current bridge status."""
    return json.dumps(_request_json("GET", "/status"))


def mumble_connect(args: Dict[str, Any]) -> str:
    """Connect the bridge to a Mumble server."""
    payload = _default_connect_payload(args)
    if not payload.get("server"):
        return json.dumps({"error": "Missing Mumble server. Pass server or set MUMBLE_SERVER / MUMBLE_BRIDGE_URL."})
    if not payload.get("username"):
        return json.dumps({"error": "Missing Mumble username. Pass username or set MUMBLE_USERNAME."})
    return json.dumps(_request_json("POST", "/connect", payload=payload))


def mumble_disconnect(args: Dict[str, Any]) -> str:
    """Disconnect the bridge from the Mumble server."""
    return json.dumps(_request_json("POST", "/disconnect"))


def mumble_message(args: Dict[str, Any]) -> str:
    """Send a text message to the current Mumble channel or a target user."""
    message = (args.get("message") or "").strip()
    if not message:
        return json.dumps({"error": "message is required"})
    payload = {
        "message": message,
        "target": (args.get("target") or "channel").strip(),
        "username": (args.get("username") or None),
    }
    return json.dumps(_request_json("POST", "/message", payload=payload))


def mumble_join(args: Dict[str, Any]) -> str:
    """Join a Mumble channel by name or path."""
    channel = (args.get("channel") or "").strip()
    if not channel:
        return json.dumps({"error": "channel is required"})
    return json.dumps(_request_json("POST", "/join", form={"channel": channel}))


def mumble_send_audio_file(args: Dict[str, Any]) -> str:
    """Upload a local WAV file to the bridge and queue it for playback."""
    path_value = args.get("path") or args.get("file") or args.get("wav_path")
    if not path_value:
        return json.dumps({"error": "path is required"})

    path = Path(path_value).expanduser()
    if not path.exists():
        return json.dumps({"error": f"Audio file not found: {path}"})

    wav_bytes = path.read_bytes()
    files = {"file": (path.name, wav_bytes, "audio/wav")}
    return json.dumps(_request_json("POST", "/audio/wav", files=files))


def _bridge_ready() -> Dict[str, Any]:
    status = _current_bridge_status()
    if status.get("error"):
        return status
    if not status.get("connected"):
        return {"error": "Mumble bridge is not connected"}
    if not status.get("ready"):
        return {"error": "Mumble bridge is connected but not ready"}
    return {"ok": True, "status": status}


def _ensure_bridge_wav(path: Path) -> Path:
    """Convert audio to 48 kHz mono PCM16 WAV when required by the bridge."""
    try:
        with wave.open(str(path), "rb") as wf:
            if (
                wf.getsampwidth() == 2
                and wf.getframerate() == 48000
                and wf.getnchannels() in (1, 2)
            ):
                return path
    except wave.Error:
        pass

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to normalize audio for the Mumble bridge")

    fd, normalized_path = tempfile.mkstemp(prefix="mumble_bridge_", suffix=".wav")
    os.close(fd)
    subprocess.run(
        [ffmpeg, "-y", "-i", str(path), "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", normalized_path],
        check=True,
        timeout=60,
        capture_output=True,
    )
    return Path(normalized_path)


def mumble_speak(args: Dict[str, Any]) -> str:
    """Generate TTS with Hermes' current voice and play it into Mumble."""
    text = (args.get("text") or args.get("message") or "").strip()
    if not text:
        return json.dumps({"error": "text is required"})

    ready = _bridge_ready()
    if ready.get("error"):
        return json.dumps(ready)

    from tools.tts_tool import text_to_speech_tool

    with tempfile.TemporaryDirectory(prefix="hermes-mumble-tts-") as tmpdir:
        raw_path = Path(tmpdir) / "speech.wav"
        tts_result = text_to_speech_tool(text, output_path=str(raw_path))
        try:
            tts_data = json.loads(tts_result)
        except Exception:
            return json.dumps({"error": f"Unexpected TTS response: {tts_result}"})

        if not tts_data.get("success"):
            return json.dumps({"error": tts_data.get("error", "TTS generation failed")})

        generated_path = Path(tts_data.get("file_path") or raw_path)
        if not generated_path.exists():
            return json.dumps({"error": f"TTS output file not found: {generated_path}"})

        normalized_path = generated_path
        temp_normalized = False
        try:
            normalized_path = _ensure_bridge_wav(generated_path)
            temp_normalized = normalized_path != generated_path
            files = {"file": (normalized_path.name, normalized_path.read_bytes(), "audio/wav")}
            bridge_result = _request_json("POST", "/audio/wav", files=files)
        finally:
            if temp_normalized:
                normalized_path.unlink(missing_ok=True)

    if bridge_result.get("error"):
        return json.dumps(bridge_result)

    return json.dumps(
        {
            "ok": True,
            "text": text,
            "bytes": bridge_result.get("bytes"),
            "bridge": ready.get("status", {}),
        }
    )


def mumble_events(args: Dict[str, Any]) -> str:
    """Return the rolling bridge event log."""
    after = args.get("after")
    try:
        after_value = int(after) if after is not None else 0
    except Exception:
        after_value = 0
    return json.dumps(_request_json("GET", f"/events?after={after_value}"))


def mumble_transcripts(args: Dict[str, Any]) -> str:
    """Return the transcript events emitted by the bridge."""
    after = args.get("after")
    try:
        after_value = int(after) if after is not None else 0
    except Exception:
        after_value = 0
    return json.dumps(_request_json("GET", f"/transcripts?after={after_value}"))


def mumble_transcribe_audio_file(args: Dict[str, Any]) -> str:
    """Upload a local WAV file to the bridge and return the transcript."""
    path_value = args.get("path") or args.get("file") or args.get("wav_path")
    if not path_value:
        return json.dumps({"error": "path is required"})

    path = Path(path_value).expanduser()
    if not path.exists():
        return json.dumps({"error": f"Audio file not found: {path}"})

    model = args.get("model") or None
    language = args.get("language") or None
    speaker = args.get("speaker") or None
    with path.open("rb") as handle:
        files = {"file": (path.name, handle.read(), "audio/wav")}
    form = {}
    if model:
        form["model"] = model
    if language:
        form["language"] = language
    if speaker:
        form["speaker"] = speaker
    return json.dumps(_request_json("POST", "/transcribe/wav", files=files, form=form or None))


def _bridge_available() -> bool:
    status = _request_json("GET", "/health")
    return bool(status.get("ok"))


MUMBLE_STATUS_SCHEMA = {
    "name": "mumble_status",
    "description": "Get the current local Mumble bridge status and connection state.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

MUMBLE_CONNECT_SCHEMA = {
    "name": "mumble_connect",
    "description": "Connect the local Mumble bridge to a server. If arguments are omitted, defaults are read from MUMBLE_SERVER, MUMBLE_PORT, and MUMBLE_USERNAME.",
    "parameters": {
        "type": "object",
        "properties": {
            "server": {"type": "string", "description": "Mumble server host or IP"},
            "port": {"type": "integer", "description": "Mumble server port"},
            "username": {"type": "string", "description": "Username to use on the server"},
            "password": {"type": "string", "description": "Server password, if required"},
            "insecure": {"type": "boolean", "description": "Skip certificate validation"},
            "channel": {"type": "string", "description": "Channel to join after connecting"},
        },
        "required": [],
    },
}

MUMBLE_DISCONNECT_SCHEMA = {
    "name": "mumble_disconnect",
    "description": "Disconnect the local Mumble bridge from the server.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

MUMBLE_MESSAGE_SCHEMA = {
    "name": "mumble_message",
    "description": "Send a text message to the current Mumble channel or to a specific user.",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Message text to send"},
            "target": {"type": "string", "description": "Target type: channel or user"},
            "username": {"type": "string", "description": "Target username when sending to a specific user"},
        },
        "required": ["message"],
    },
}

MUMBLE_JOIN_SCHEMA = {
    "name": "mumble_join",
    "description": "Join a Mumble channel by name or path.",
    "parameters": {
        "type": "object",
        "properties": {
            "channel": {"type": "string", "description": "Channel name or full path"},
        },
        "required": ["channel"],
    },
}

MUMBLE_SEND_AUDIO_FILE_SCHEMA = {
    "name": "mumble_send_audio_file",
    "description": "Upload a local WAV file to the bridge and queue it for playback in Mumble.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to a local WAV file"},
        },
        "required": ["path"],
    },
}

MUMBLE_SPEAK_SCHEMA = {
    "name": "mumble_speak",
    "description": "Generate speech using Hermes' configured TTS voice and immediately play it in Mumble through the local bridge.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to speak in Mumble"},
        },
        "required": ["text"],
    },
}

MUMBLE_EVENTS_SCHEMA = {
    "name": "mumble_events",
    "description": "Fetch recent Mumble bridge events for debugging or monitoring.",
    "parameters": {
        "type": "object",
        "properties": {
            "after": {"type": "integer", "description": "Only return events after this event id"},
        },
        "required": [],
    },
}

MUMBLE_TRANSCRIPTS_SCHEMA = {
    "name": "mumble_transcripts",
    "description": "Fetch transcript events emitted by the bridge.",
    "parameters": {
        "type": "object",
        "properties": {
            "after": {"type": "integer", "description": "Only return transcript events after this event id"},
        },
        "required": [],
    },
}

MUMBLE_TRANSCRIBE_AUDIO_FILE_SCHEMA = {
    "name": "mumble_transcribe_audio_file",
    "description": "Upload a local WAV file to the bridge and get a transcript back.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to a local WAV file"},
            "model": {"type": "string", "description": "Whisper model to use (small, medium, etc.)"},
            "language": {"type": "string", "description": "Optional language code"},
            "speaker": {"type": "string", "description": "Optional speaker label"},
        },
        "required": ["path"],
    },
}

registry.register(
    name="mumble_status",
    toolset="mumble",
    schema=MUMBLE_STATUS_SCHEMA,
    handler=lambda args, **kw: mumble_status(args),
    check_fn=_bridge_available,
    description="Query the local Mumble bridge status.",
    emoji="🎤",
)
registry.register(
    name="mumble_connect",
    toolset="mumble",
    schema=MUMBLE_CONNECT_SCHEMA,
    handler=lambda args, **kw: mumble_connect(args),
    check_fn=_bridge_available,
    description="Connect the local Mumble bridge to a server.",
    emoji="🔌",
)
registry.register(
    name="mumble_disconnect",
    toolset="mumble",
    schema=MUMBLE_DISCONNECT_SCHEMA,
    handler=lambda args, **kw: mumble_disconnect(args),
    check_fn=_bridge_available,
    description="Disconnect the local Mumble bridge.",
    emoji="⏏️",
)
registry.register(
    name="mumble_message",
    toolset="mumble",
    schema=MUMBLE_MESSAGE_SCHEMA,
    handler=lambda args, **kw: mumble_message(args),
    check_fn=_bridge_available,
    description="Send a text message to Mumble.",
    emoji="💬",
)
registry.register(
    name="mumble_join",
    toolset="mumble",
    schema=MUMBLE_JOIN_SCHEMA,
    handler=lambda args, **kw: mumble_join(args),
    check_fn=_bridge_available,
    description="Join a Mumble channel.",
    emoji="🚪",
)
registry.register(
    name="mumble_send_audio_file",
    toolset="mumble",
    schema=MUMBLE_SEND_AUDIO_FILE_SCHEMA,
    handler=lambda args, **kw: mumble_send_audio_file(args),
    check_fn=_bridge_available,
    description="Upload a WAV file for playback in Mumble.",
    emoji="🔊",
)
registry.register(
    name="mumble_speak",
    toolset="mumble",
    schema=MUMBLE_SPEAK_SCHEMA,
    handler=lambda args, **kw: mumble_speak(args),
    check_fn=_bridge_available,
    description="Generate TTS and speak it in Mumble.",
    emoji="🗣️",
)
registry.register(
    name="mumble_events",
    toolset="mumble",
    schema=MUMBLE_EVENTS_SCHEMA,
    handler=lambda args, **kw: mumble_events(args),
    check_fn=_bridge_available,
    description="Fetch Mumble bridge events.",
    emoji="🧾",
)
registry.register(
    name="mumble_transcripts",
    toolset="mumble",
    schema=MUMBLE_TRANSCRIPTS_SCHEMA,
    handler=lambda args, **kw: mumble_transcripts(args),
    check_fn=_bridge_available,
    description="Fetch transcript events from the bridge.",
    emoji="📝",
)
registry.register(
    name="mumble_transcribe_audio_file",
    toolset="mumble",
    schema=MUMBLE_TRANSCRIBE_AUDIO_FILE_SCHEMA,
    handler=lambda args, **kw: mumble_transcribe_audio_file(args),
    check_fn=_bridge_available,
    description="Upload a WAV file and get a transcript back.",
    emoji="🎙️",
)
