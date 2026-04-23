"""Mumble platform adapter backed by the local hermes-mumble-bridge.

Inbound flow:
  bridge /events -> transcript_received -> MessageEvent -> normal Hermes session

Outbound flow:
  Hermes text reply -> bridge /message
  Hermes voice reply -> bridge /audio/wav
"""

from __future__ import annotations

import asyncio
import audioop
import io
import logging
import os
import shutil
import subprocess
import tempfile
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via check_mumble_requirements
    httpx = None  # type: ignore[assignment]
    HTTPX_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult

logger = logging.getLogger(__name__)

DEFAULT_BRIDGE_URL = "http://127.0.0.1:8789"
DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 15.0
STATUS_REFRESH_SECONDS = 30.0
PCM_SAMPLE_RATE = 48_000
PCM_SAMPLE_WIDTH = 2


def check_mumble_requirements() -> bool:
    """Check whether the Mumble adapter can run."""
    return HTTPX_AVAILABLE


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return default


class MumbleAdapter(BasePlatformAdapter):
    """Gateway adapter that polls a local Mumble bridge for transcripts."""
    SUPPORTS_MESSAGE_EDITING = False
    ALLOW_TOOL_PROGRESS_WITHOUT_EDITING = True

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.MUMBLE)
        extra = config.extra if isinstance(config.extra, dict) else {}
        self._bridge_base_url = str(
            extra.get("base_url") or os.getenv("MUMBLE_BRIDGE_URL", DEFAULT_BRIDGE_URL)
        ).strip().rstrip("/")
        self._poll_interval_seconds = float(
            extra.get("poll_interval_seconds", DEFAULT_POLL_INTERVAL_SECONDS)
        )
        self._request_timeout_seconds = float(
            extra.get("request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS)
        )
        self._replay_existing_events = _coerce_bool(
            extra.get("replay_existing_events"),
            False,
        )
        self._client: Optional["httpx.AsyncClient"] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._next_event_id = 0
        self._bridge_username: Optional[str] = None
        self._server_name: Optional[str] = None
        self._server_port: Optional[int] = None
        self._channel_name: Optional[str] = None
        self._channel_id: Optional[str] = None
        self._users_by_session: Dict[str, str] = {}
        self._status_refreshed_at = 0.0

    async def connect(self) -> bool:
        if not HTTPX_AVAILABLE:
            logger.warning("[mumble] httpx is not installed")
            return False
        if not self._bridge_base_url:
            logger.error("[mumble] No bridge URL configured")
            return False

        self._client = httpx.AsyncClient(
            base_url=self._bridge_base_url,
            timeout=self._request_timeout_seconds,
        )
        try:
            await self._refresh_status(force=True)
            if not self._replay_existing_events:
                await self._prime_event_cursor()
        except Exception as exc:
            logger.error("[mumble] Failed to connect to bridge %s: %s", self._bridge_base_url, exc)
            await self._close_client()
            return False

        self._mark_connected()
        self._poll_task = asyncio.create_task(self._poll_loop(), name="mumble-poll")
        logger.info(
            "[mumble] Connected to bridge %s as %s in %s",
            self._bridge_base_url,
            self._bridge_username or "(unknown)",
            self._channel_name or "(unknown channel)",
        )
        return True

    async def disconnect(self) -> None:
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._poll_task = None
        await self._close_client()
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        del chat_id, reply_to, metadata
        client = self._require_client()
        try:
            response = await client.post(
                "/message",
                json={"message": content, "target": "channel"},
            )
            response.raise_for_status()
            payload = response.json()
            return SendResult(
                success=bool(payload.get("ok")),
                raw_response=payload,
            )
        except Exception as exc:
            return SendResult(
                success=False,
                error=str(exc),
                retryable=self._is_retryable_exception(exc),
            )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        del chat_id, caption, reply_to, kwargs
        client = self._require_client()
        try:
            wav_bytes = await asyncio.to_thread(self._prepare_wav_bytes, audio_path)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

        try:
            response = await client.post(
                "/audio/wav",
                files={"file": ("reply.wav", wav_bytes, "audio/wav")},
            )
            response.raise_for_status()
            payload = response.json()
            return SendResult(
                success=bool(payload.get("ok")),
                raw_response=payload,
            )
        except Exception as exc:
            return SendResult(
                success=False,
                error=str(exc),
                retryable=self._is_retryable_exception(exc),
            )

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        del chat_id, metadata
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        del chat_id
        await self._refresh_status(force=True)
        return {
            "name": self._channel_name or self._server_chat_id(),
            "type": "channel",
            "chat_id": self._server_chat_id(),
        }

    def _require_client(self) -> "httpx.AsyncClient":
        if self._client is None:
            raise RuntimeError("Mumble bridge client is not connected")
        return self._client

    async def _close_client(self) -> None:
        if self._client is not None:
            await self._client.aclose()
        self._client = None

    async def _refresh_status(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._status_refreshed_at) < STATUS_REFRESH_SECONDS:
            return
        client = self._require_client()
        response = await client.get("/status")
        response.raise_for_status()
        payload = response.json()
        self._bridge_username = str(payload.get("username") or "").strip() or None
        self._server_name = str(payload.get("server") or "").strip() or None
        port = payload.get("port")
        self._server_port = int(port) if isinstance(port, int) else None
        self._channel_name = str(payload.get("channel") or "").strip() or None
        channel_id = payload.get("channel_id")
        self._channel_id = str(channel_id) if channel_id is not None else None
        users = payload.get("users")
        users_by_session: Dict[str, str] = {}
        if isinstance(users, list):
            for user in users:
                if not isinstance(user, dict):
                    continue
                session = user.get("session")
                name = str(user.get("name") or "").strip()
                if session is None or not name:
                    continue
                users_by_session[str(session)] = name
        self._users_by_session = users_by_session
        self._status_refreshed_at = now

    async def _prime_event_cursor(self) -> None:
        events = await self._fetch_events(after=0)
        if events:
            self._next_event_id = max(int(event.get("id", 0)) for event in events)

    async def _poll_loop(self) -> None:
        failure_delay = self._poll_interval_seconds
        while True:
            try:
                events = await self._fetch_events(after=self._next_event_id)
                for raw_event in events:
                    event_id = int(raw_event.get("id", 0))
                    if event_id > self._next_event_id:
                        self._next_event_id = event_id
                    await self._handle_bridge_event(raw_event)
                failure_delay = self._poll_interval_seconds
                await asyncio.sleep(self._poll_interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[mumble] Poll error: %s", exc)
                await asyncio.sleep(failure_delay)
                failure_delay = min(failure_delay * 2, 30.0)

    async def _fetch_events(self, *, after: int) -> list[dict[str, Any]]:
        client = self._require_client()
        response = await client.get("/events", params={"after": after})
        response.raise_for_status()
        payload = response.json()
        events = payload.get("events")
        return events if isinstance(events, list) else []

    async def _handle_bridge_event(self, raw_event: dict[str, Any]) -> None:
        kind = str(raw_event.get("kind") or "").strip()
        if kind == "transcript_received":
            await self._handle_transcript_event(raw_event)
        elif kind == "text_message":
            await self._handle_text_message_event(raw_event)
        elif kind in {"connected", "channel_joined"}:
            await self._refresh_status(force=True)

    async def _handle_transcript_event(self, raw_event: dict[str, Any]) -> None:
        data = raw_event.get("data")
        if not isinstance(data, dict):
            return
        transcript = str(data.get("transcript") or "").strip()
        speaker = str(data.get("speaker") or "").strip()
        if not transcript or not speaker:
            return

        await self._refresh_status()
        if self._bridge_username and speaker.casefold() == self._bridge_username.casefold():
            logger.debug("[mumble] Ignoring self transcript from %s", speaker)
            return

        raw_ts = raw_event.get("ts")
        if isinstance(raw_ts, (int, float)):
            timestamp = datetime.fromtimestamp(raw_ts, tz=timezone.utc)
        else:
            timestamp = datetime.now(tz=timezone.utc)

        event = MessageEvent(
            text=transcript,
            message_type=MessageType.VOICE,
            source=self.build_source(
                chat_id=self._server_chat_id(),
                chat_name=self._server_chat_id(),
                chat_type="channel",
                user_id=speaker,
                user_name=speaker,
                thread_id=self._thread_id(),
                chat_topic=self._channel_name,
            ),
            raw_message=raw_event,
            message_id=str(raw_event.get("id")) if raw_event.get("id") is not None else None,
            timestamp=timestamp,
        )
        await self.handle_message(event)

    async def _handle_text_message_event(self, raw_event: dict[str, Any]) -> None:
        data = raw_event.get("data")
        if not isinstance(data, dict):
            return

        text = str(data.get("text") or "").strip()
        if not text:
            return

        await self._refresh_status()
        actor = data.get("actor")
        actor_session = str(actor) if actor is not None else ""
        speaker = self._users_by_session.get(actor_session) or actor_session or "unknown"
        if self._bridge_username and speaker.casefold() == self._bridge_username.casefold():
            logger.debug("[mumble] Ignoring self text message from %s", speaker)
            return

        raw_ts = raw_event.get("ts")
        if isinstance(raw_ts, (int, float)):
            timestamp = datetime.fromtimestamp(raw_ts, tz=timezone.utc)
        else:
            timestamp = datetime.now(tz=timezone.utc)

        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=self.build_source(
                chat_id=self._server_chat_id(),
                chat_name=self._server_chat_id(),
                chat_type="channel",
                user_id=speaker,
                user_name=speaker,
                thread_id=self._thread_id(),
                chat_topic=self._channel_name,
            ),
            raw_message=raw_event,
            message_id=str(raw_event.get("id")) if raw_event.get("id") is not None else None,
            timestamp=timestamp,
        )
        await self.handle_message(event)

    def _server_chat_id(self) -> str:
        if self._server_name and self._server_port is not None:
            return f"{self._server_name}:{self._server_port}"
        if self._server_name:
            return self._server_name
        return self._bridge_base_url

    def _thread_id(self) -> Optional[str]:
        return self._channel_id or self._channel_name

    def _prepare_wav_bytes(self, audio_path: str) -> bytes:
        path = Path(audio_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if path.suffix.lower() == ".wav":
            return self._normalize_wav_bytes(path.read_bytes())
        return self._convert_audio_file_to_wav_bytes(path)

    def _convert_audio_file_to_wav_bytes(self, path: Path) -> bytes:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                f"ffmpeg is required to convert non-WAV audio for Mumble: {path}"
            )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            result = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(path),
                    "-ac",
                    "1",
                    "-ar",
                    str(PCM_SAMPLE_RATE),
                    "-sample_fmt",
                    "s16",
                    str(tmp_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"ffmpeg conversion failed: {stderr}")
            return tmp_path.read_bytes()
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    def _normalize_wav_bytes(self, wav_bytes: bytes) -> bytes:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            pcm = wf.readframes(wf.getnframes())

        if sample_width != PCM_SAMPLE_WIDTH:
            pcm = audioop.lin2lin(pcm, sample_width, PCM_SAMPLE_WIDTH)
            sample_width = PCM_SAMPLE_WIDTH
        if channels == 2:
            pcm = audioop.tomono(pcm, sample_width, 0.5, 0.5)
            channels = 1
        if channels != 1:
            raise RuntimeError(f"Unsupported WAV channel count for Mumble: {channels}")
        if frame_rate != PCM_SAMPLE_RATE:
            pcm, _ = audioop.ratecv(
                pcm,
                sample_width,
                channels,
                frame_rate,
                PCM_SAMPLE_RATE,
                None,
            )

        bio = io.BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(PCM_SAMPLE_WIDTH)
            wf.setframerate(PCM_SAMPLE_RATE)
            wf.writeframes(pcm)
        return bio.getvalue()

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        if not HTTPX_AVAILABLE:
            return False
        return isinstance(
            exc,
            (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.NetworkError,
                httpx.PoolTimeout,
            ),
        )
