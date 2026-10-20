"""Tests for energiedelen (energy sharing) support."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest

from custom_components.fluvius.api import (
    ENERGIEDELEN_API_BASE,
    FluviusApiClient,
)
from custom_components.fluvius.const import METER_TYPE_GAS


class DummyResponse:
    def __init__(self, status: int = 200, payload=None) -> None:
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    async def json(self):
        return self._payload


class FakeSession:
    """Routes energiedelen API calls to canned responses and records them."""

    def __init__(self, overview_status: int = 200, overview_payload=None) -> None:
        self.overview_status = overview_status
        self.overview_payload = overview_payload
        self.chart_payloads: dict[int, object] = {}
        self.calls: list[dict] = []

    def request(self, method, url, *, json=None, headers=None, timeout=None):
        self.calls.append({"method": method, "url": url, "json": json, "headers": headers})
        if url == f"{ENERGIEDELEN_API_BASE}/energy-exchanges":
            return DummyResponse(self.overview_status, self.overview_payload)
        if "/chart-exchanged-energy" in url:
            period_type = (json or {}).get("chartPeriodType")
            return DummyResponse(200, self.chart_payloads.get(period_type, []))
        return DummyResponse(404, None)


def _make_client(session: FakeSession, *, meter_type: str | None = None) -> FluviusApiClient:
    kwargs = {"meter_type": meter_type} if meter_type else {}
    client = FluviusApiClient(
        session=session,  # type: ignore[arg-type]
        email="user@example.com",
        password="secret",
        ean="541448800000000000",
        meter_serial="1SAGTEST",
        options={},
        **kwargs,
    )
    client._access_token = "cached-token"  # pylint: disable=protected-access
    client._token_expires = time.monotonic() + 3600  # pylint: disable=protected-access
    return client


def _freeze_now(monkeypatch, fixed_now: datetime) -> None:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            if tz:
                return fixed_now.astimezone(tz)
            return fixed_now

    monkeypatch.setattr("custom_components.fluvius.api.datetime", FixedDateTime)


async def test_no_energy_sharing_returns_empty_list():
    """Accounts without energiedelen get an empty list and no chart calls."""

    session = FakeSession(overview_payload={"data": [], "page": 1, "totalItems": 0})
    client = _make_client(session)

    assert await client.fetch_energy_sharing() == []
    assert len(session.calls) == 1
    assert session.calls[0]["method"] == "GET"
    assert session.calls[0]["headers"]["Authorization"] == "Bearer cached-token"


async def test_gas_meter_never_calls_energiedelen():
    session = FakeSession(overview_payload={"data": [{"id": 1}]})
    client = _make_client(session, meter_type=METER_TYPE_GAS)

    assert await client.fetch_energy_sharing() == []
    assert session.calls == []


async def test_unavailable_feature_returns_empty_list():
    session = FakeSession(overview_status=404)
    client = _make_client(session)

    assert await client.fetch_energy_sharing() == []


async def test_summary_parsing_and_totals(monkeypatch):
    """One exchange: chart calls use the SPA contract; totals split day/month/year."""

    _freeze_now(monkeypatch, datetime(2026, 10, 20, 10, 0, tzinfo=UTC))
    session = FakeSession(
        overview_payload={"data": [{"id": 42, "name": "Vriendengroep"}], "totalItems": 1}
    )
    session.chart_payloads[6] = {  # month view -> day buckets
        "unit": "kWh",
        "exchangedEnergy": [
            {  # today (local 2026-10-20 starts 2026-10-19T22:00Z)
                "startDateUtc": "2026-10-19T22:00:00Z",
                "measurement": {"consumptionValue": 1.5, "injectionValue": 0.5},
            },
            {
                "startDateUtc": "2026-10-18T22:00:00Z",
                "measurement": {"consumptionValue": 2.0, "injectionValue": 1.0},
            },
            {
                "startDateUtc": "2026-10-04T22:00:00Z",
                "measurement": {"consumptionValue": 3.0, "injectionValue": 0.0},
            },
            {  # September: excluded from month totals
                "startDateUtc": "2026-09-29T22:00:00Z",
                "measurement": {"consumptionValue": 9.9, "injectionValue": 9.9},
            },
        ],
    }
    session.chart_payloads[7] = {  # year view -> month buckets
        "unit": "kWh",
        "exchangedEnergy": [
            {
                "startDateUtc": "2026-09-30T22:00:00Z",
                "measurement": {"consumptionValue": 10.0, "injectionValue": 4.0},
            },
            {
                "startDateUtc": "2026-04-30T22:00:00Z",
                "measurement": {"consumptionValue": 20.0, "injectionValue": 6.0},
            },
            {  # last year: excluded from year totals
                "startDateUtc": "2025-11-30T23:00:00Z",
                "measurement": {"consumptionValue": 99.0, "injectionValue": 99.0},
            },
        ],
    }
    client = _make_client(session)

    summaries = await client.fetch_energy_sharing()

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.exchange_id == 42
    assert summary.name == "Vriendengroep"
    assert len(summary.daily) == 4
    assert len(summary.monthly) == 3

    chart_calls = [c for c in session.calls if c["method"] == "POST"]
    assert len(chart_calls) == 2
    bodies = sorted(chart_calls, key=lambda c: c["json"]["chartPeriodType"])
    assert bodies[0]["url"] == f"{ENERGIEDELEN_API_BASE}/energy-exchanges/42/chart-exchanged-energy"
    assert bodies[0]["json"]["chartPeriodType"] == 6
    assert bodies[0]["json"]["eans"] == ["541448800000000000"]
    assert bodies[0]["json"]["energyExchangeId"] == 42
    assert bodies[1]["json"]["chartPeriodType"] == 7

    totals = summary.totals
    assert totals["received_today"] == pytest.approx(1.5)
    assert totals["injected_today"] == pytest.approx(0.5)
    assert totals["received_month"] == pytest.approx(1.5 + 2.0 + 3.0)
    assert totals["injected_month"] == pytest.approx(0.5 + 1.0 + 0.0)
    assert totals["received_year"] == pytest.approx(10.0 + 20.0)
    assert totals["injected_year"] == pytest.approx(4.0 + 6.0)


async def test_malformed_items_are_dropped():
    session = FakeSession(overview_payload={"data": [{"id": 7}]})
    session.chart_payloads[6] = [
        {"startDateUtc": "2026-10-19T22:00:00Z"},  # no measurement values
        "garbage",
        {
            "startDateUtc": "2026-10-19T22:00:00Z",
            "measurement": {"consumptionValue": 1.0},
        },
    ]
    session.chart_payloads[7] = []
    client = _make_client(session)

    summaries = await client.fetch_energy_sharing()

    assert len(summaries) == 1
    assert len(summaries[0].daily) == 1
    assert summaries[0].daily[0].consumption == pytest.approx(1.0)
    assert summaries[0].daily[0].injection == 0.0


async def test_overview_items_without_id_are_skipped():
    session = FakeSession(overview_payload={"data": [{"name": "no-id-here"}]})
    client = _make_client(session)

    assert await client.fetch_energy_sharing() == []
    assert all("/chart-exchanged-energy" not in c["url"] for c in session.calls)
