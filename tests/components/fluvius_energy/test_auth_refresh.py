"""Tests for the Fluvius token refresh and authentication error mapping."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import aiohttp
import pytest

from custom_components.fluvius.api import (
    FluviusApiClient,
    FluviusApiError,
    FluviusAuthenticationError,
)
from custom_components.fluvius.auth import (
    FluviusAuthError,
    FluviusCredentialsError,
    FluviusInvalidRefreshError,
)


class FakeTokenStore:
    """In-memory stand-in for the HA Store-backed token store."""

    def __init__(self, data: dict | None = None) -> None:
        self.data = data
        self.saves: list[dict] = []

    async def async_load(self) -> dict | None:
        return self.data

    async def async_save(self, data: dict) -> None:
        self.saves.append(data)
        self.data = data


def _make_client(*, token_store: FakeTokenStore | None = None) -> FluviusApiClient:
    session = MagicMock(spec=aiohttp.ClientSession)
    return FluviusApiClient(
        session=session,
        email="user@example.com",
        password="secret",
        ean="541448800000000000",
        meter_serial="1SAGTEST",
        options={},
        token_store=token_store,
    )


def _patch_auth(monkeypatch, *, refresh=None, login=None):
    """Patch both token helpers in the api module namespace."""

    async def fail_refresh(*_args, **_kwargs):
        pytest.fail("async_refresh_bearer_token should not have been called")

    async def fail_login(*_args, **_kwargs):
        pytest.fail("async_get_bearer_token should not have been called")

    monkeypatch.setattr(
        "custom_components.fluvius.api.async_refresh_bearer_token", refresh or fail_refresh
    )
    monkeypatch.setattr("custom_components.fluvius.api.async_get_bearer_token", login or fail_login)


async def test_refresh_token_used_when_access_token_expired(monkeypatch):
    """An expired access token must be renewed via the refresh grant, not a full login."""

    store = FakeTokenStore()
    client = _make_client(token_store=store)
    client._refresh_token = "rt-1"  # pylint: disable=protected-access

    async def fake_refresh(_session, refresh_token, *, verbose=False):
        assert refresh_token == "rt-1"
        return "new-access", {
            "access_token": "new-access",
            "refresh_token": "rt-2",
            "expires_in": 3600,
        }

    _patch_auth(monkeypatch, refresh=fake_refresh)

    token = await client._async_get_access_token()  # pylint: disable=protected-access

    assert token == "new-access"
    assert client._refresh_token == "rt-2"  # pylint: disable=protected-access
    assert store.saves[-1]["refresh_token"] == "rt-2"
    assert store.saves[-1]["access_token"] == "new-access"


async def test_invalid_refresh_falls_back_to_full_login(monkeypatch):
    """A rejected refresh token (invalid_grant) triggers one full credential login."""

    store = FakeTokenStore()
    client = _make_client(token_store=store)
    client._refresh_token = "rt-dead"  # pylint: disable=protected-access

    async def fake_refresh(*_args, **_kwargs):
        raise FluviusInvalidRefreshError("invalid_grant")

    async def fake_login(_session, email, password, *, remember_me=False, verbose=False):
        assert email == "user@example.com"
        return "full-access", {
            "access_token": "full-access",
            "refresh_token": "rt-3",
            "expires_in": 3600,
        }

    _patch_auth(monkeypatch, refresh=fake_refresh, login=fake_login)

    token = await client._async_get_access_token()  # pylint: disable=protected-access

    assert token == "full-access"
    assert client._refresh_token == "rt-3"  # pylint: disable=protected-access
    assert store.saves[-1]["refresh_token"] == "rt-3"


async def test_transient_refresh_failure_is_not_an_auth_failure(monkeypatch):
    """A transient refresh error (WAF/5xx) must NOT raise FluviusAuthenticationError."""

    client = _make_client()
    client._refresh_token = "rt-1"  # pylint: disable=protected-access

    async def fake_refresh(*_args, **_kwargs):
        raise FluviusAuthError("Token endpoint error during refresh (503)")

    _patch_auth(monkeypatch, refresh=fake_refresh)

    with pytest.raises(FluviusApiError) as excinfo:
        await client._async_get_access_token()  # pylint: disable=protected-access
    assert not isinstance(excinfo.value, FluviusAuthenticationError)


async def test_full_login_credentials_error_raises_authentication_error(monkeypatch):
    """Only an explicit credential rejection may trigger reauthentication."""

    client = _make_client()

    async def fake_login(*_args, **_kwargs):
        raise FluviusCredentialsError("Credential submission failed")

    _patch_auth(monkeypatch, login=fake_login)

    with pytest.raises(FluviusAuthenticationError):
        await client._async_get_access_token()  # pylint: disable=protected-access


async def test_full_login_transient_error_raises_api_error(monkeypatch):
    """A scraping/endpoint hiccup during full login is transient, not a reauth trigger."""

    client = _make_client()

    async def fake_login(*_args, **_kwargs):
        raise FluviusAuthError("Unable to locate `SETTINGS` payload")

    _patch_auth(monkeypatch, login=fake_login)

    with pytest.raises(FluviusApiError) as excinfo:
        await client._async_get_access_token()  # pylint: disable=protected-access
    assert not isinstance(excinfo.value, FluviusAuthenticationError)


async def test_valid_stored_tokens_avoid_any_http(monkeypatch):
    """Stored tokens from a previous run are reused while still valid."""

    store = FakeTokenStore(
        {
            "refresh_token": "rt-9",
            "access_token": "stored-access",
            "access_expires_at": time.time() + 3600,
        }
    )
    client = _make_client(token_store=store)
    _patch_auth(monkeypatch)

    token = await client._async_get_access_token()  # pylint: disable=protected-access

    assert token == "stored-access"
    assert client._refresh_token == "rt-9"  # pylint: disable=protected-access


async def test_expired_stored_access_token_uses_stored_refresh(monkeypatch):
    """An expired stored access token falls through to the stored refresh token."""

    store = FakeTokenStore(
        {
            "refresh_token": "rt-9",
            "access_token": "stored-access",
            "access_expires_at": time.time() - 10,
        }
    )
    client = _make_client(token_store=store)

    async def fake_refresh(_session, refresh_token, *, verbose=False):
        assert refresh_token == "rt-9"
        return "refreshed-access", {"access_token": "refreshed-access", "expires_in": 3600}

    _patch_auth(monkeypatch, refresh=fake_refresh)

    token = await client._async_get_access_token()  # pylint: disable=protected-access

    assert token == "refreshed-access"
    # No new refresh token in the response: the old one is kept.
    assert client._refresh_token == "rt-9"  # pylint: disable=protected-access
