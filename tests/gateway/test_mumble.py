"""Tests for the Mumble gateway adapter."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, *, status_payload=None, post_payload=None):
        self.status_payload = status_payload or {}
        self.post_payload = post_payload or {"ok": True}
        self.get_calls = []
        self.post_calls = []

    async def get(self, path, params=None):
        self.get_calls.append({"path": path, "params": params})
        if path == "/status":
            return _FakeResponse(self.status_payload)
        if path == "/events":
            return _FakeResponse({"events": []})
        raise AssertionError(f"Unexpected GET {path}")

    async def post(self, path, json=None, files=None):
        self.post_calls.append({"path": path, "json": json, "files": files})
        return _FakeResponse(self.post_payload)

    async def aclose(self):
        return None


class TestMumbleConfigLoading:
    def test_apply_env_overrides_mumble(self, monkeypatch):
        monkeypatch.setenv("MUMBLE_BRIDGE_URL", "http://127.0.0.1:8789")
        monkeypatch.setenv("MUMBLE_POLL_INTERVAL_SECONDS", "0.5")

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.MUMBLE in config.platforms
        mumble = config.platforms[Platform.MUMBLE]
        assert mumble.enabled is True
        assert mumble.extra["base_url"] == "http://127.0.0.1:8789"
        assert mumble.extra["poll_interval_seconds"] == 0.5
        assert Platform.MUMBLE in config.get_connected_platforms()


@pytest.mark.asyncio
class TestMumbleAdapter:
    async def test_transcript_event_builds_shared_channel_source(self):
        from gateway.platforms.mumble import MumbleAdapter

        adapter = MumbleAdapter(
            PlatformConfig(
                enabled=True,
                extra={"base_url": "http://127.0.0.1:8789"},
            )
        )
        adapter._client = _FakeClient(
            status_payload={
                "username": "BotBeta",
                "server": "10.42.0.1",
                "port": 64738,
                "channel": "Root",
                "channel_id": 0,
                "users": [{"session": 12, "name": "BotAlpha"}],
            }
        )
        adapter.handle_message = AsyncMock()

        await adapter._handle_bridge_event(
            {
                "id": 42,
                "ts": 1_700_000_000.0,
                "kind": "transcript_received",
                "data": {
                    "speaker": "BotAlpha",
                    "transcript": "hello from mumble",
                },
            }
        )

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "hello from mumble"
        assert event.message_id == "42"
        assert event.message_type.value == "voice"
        assert event.source.platform == Platform.MUMBLE
        assert event.source.chat_id == "10.42.0.1:64738"
        assert event.source.thread_id == "0"
        assert event.source.chat_name == "10.42.0.1:64738"
        assert event.source.chat_topic == "Root"
        assert event.source.chat_type == "channel"
        assert event.source.user_id == "BotAlpha"
        assert event.source.user_name == "BotAlpha"

    async def test_transcript_event_ignores_bridge_self_audio(self):
        from gateway.platforms.mumble import MumbleAdapter

        adapter = MumbleAdapter(
            PlatformConfig(
                enabled=True,
                extra={"base_url": "http://127.0.0.1:8789"},
            )
        )
        adapter._client = _FakeClient(
            status_payload={
                "username": "BotBeta",
                "server": "10.42.0.1",
                "port": 64738,
                "channel": "Root",
                "channel_id": 0,
                "users": [{"session": 24, "name": "BotBeta"}],
            }
        )
        adapter.handle_message = AsyncMock()

        await adapter._handle_bridge_event(
            {
                "id": 43,
                "kind": "transcript_received",
                "data": {
                    "speaker": "BotBeta",
                    "transcript": "this should be ignored",
                },
            }
        )

        adapter.handle_message.assert_not_awaited()

    async def test_text_message_event_builds_text_event(self):
        from gateway.platforms.mumble import MumbleAdapter

        adapter = MumbleAdapter(
            PlatformConfig(
                enabled=True,
                extra={"base_url": "http://127.0.0.1:8789"},
            )
        )
        adapter._client = _FakeClient(
            status_payload={
                "username": "BotBeta",
                "server": "10.42.0.1",
                "port": 64738,
                "channel": "Root",
                "channel_id": 0,
                "users": [{"session": 12, "name": "BotAlpha"}],
            }
        )
        adapter.handle_message = AsyncMock()

        await adapter._handle_bridge_event(
            {
                "id": 44,
                "kind": "text_message",
                "data": {
                    "text": "hello from channel chat",
                    "actor": 12,
                },
            }
        )

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "hello from channel chat"
        assert event.message_type.value == "text"
        assert event.source.platform == Platform.MUMBLE
        assert event.source.chat_id == "10.42.0.1:64738"
        assert event.source.thread_id == "0"
        assert event.source.chat_name == "10.42.0.1:64738"
        assert event.source.chat_topic == "Root"
        assert event.source.user_id == "BotAlpha"
        assert event.source.user_name == "BotAlpha"

    async def test_send_posts_channel_message_to_bridge(self):
        from gateway.platforms.mumble import MumbleAdapter

        adapter = MumbleAdapter(
            PlatformConfig(
                enabled=True,
                extra={"base_url": "http://127.0.0.1:8789"},
            )
        )
        adapter._client = _FakeClient(post_payload={"ok": True})

        result = await adapter.send(chat_id="ignored", content="reply text")

        assert result.success is True
        assert adapter._client.post_calls == [
            {
                "path": "/message",
                "json": {"message": "reply text", "target": "channel"},
                "files": None,
            }
        ]
