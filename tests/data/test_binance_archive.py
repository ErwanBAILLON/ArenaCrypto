"""The archive adapter's logic, without touching the network."""

from __future__ import annotations

import datetime as dt
import io
import zipfile

import pandas as pd

from arena.data import binance_archive as archive


class _FakeResponse:
    def __init__(self, status_code=200, content=b"", text=""):
        self.status_code = status_code
        self.content = content
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def _zip(rows: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data.csv", rows)
    return buf.getvalue()


class _Client:
    def __init__(self, responses: dict):
        self.responses = responses
        self.seen: list[str] = []

    def get(self, url, params=None):
        self.seen.append(url)
        if params:  # the listing API
            return self.responses.get("listing", _FakeResponse(text="<ListBucketResult></ListBucketResult>"))
        return self.responses.get(url, _FakeResponse(status_code=404))


class TestMonths:
    def test_it_spans_year_boundaries(self):
        out = archive._months(dt.date(2023, 11, 1), dt.date(2024, 2, 15))
        assert out == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]

    def test_a_single_month(self):
        assert archive._months(dt.date(2024, 5, 3), dt.date(2024, 5, 28)) == [(2024, 5)]


class TestKlineParsing:
    URL = f"{archive.ARCHIVE}/data/futures/um/monthly/klines/X/1h/X-1h-2024-01.zip"

    def _rows(self):
        # open_time(ms), o, h, l, c, v, close_time, ...
        return "\n".join(
            f"{1704067200000 + i * 3600000},10,11,9,{10 + i},100,{1704070799999 + i * 3600000},,,,," for i in range(3)
        )

    def test_a_missing_month_is_empty_not_an_error(self):
        client = _Client({})
        assert archive.monthly_klines(client, "X", 2024, 1).empty

    def test_bars_are_labelled_by_close_time(self):
        client = _Client({self.URL: _FakeResponse(content=_zip(self._rows()))})
        frame = archive.monthly_klines(client, "X", 2024, 1)
        assert len(frame) == 3
        # open_time 2024-01-01T00:00 + 1h = labelled 01:00, like the REST adapter
        assert frame["ts"].iloc[0] == pd.Timestamp("2024-01-01T01:00:00Z")
        assert list(frame["close"]) == [10.0, 11.0, 12.0]

    def test_a_header_row_is_skipped(self):
        rows = "open_time,open,high,low,close,volume,close_time\n" + self._rows()
        client = _Client({self.URL: _FakeResponse(content=_zip(rows))})
        assert len(archive.monthly_klines(client, "X", 2024, 1)) == 3

    def test_microsecond_stamps_are_normalised(self):
        """Archives from 2025 onward stamp microseconds; mixing units shifts bars by decades."""
        rows = f"{1704067200000000},10,11,9,10,100,{1704070799999000},,,,,"
        client = _Client({self.URL: _FakeResponse(content=_zip(rows))})
        frame = archive.monthly_klines(client, "X", 2024, 1)
        assert frame["ts"].iloc[0] == pd.Timestamp("2024-01-01T01:00:00Z")


class TestFundingParsing:
    URL = f"{archive.ARCHIVE}/data/futures/um/monthly/fundingRate/X/X-fundingRate-2024-01.zip"

    def test_rates_are_normalised_to_an_eight_hour_equivalent(self):
        """Binance moved symbols from 8h to 4h funding; averaging both is a 2x error."""
        rows = "calc_time,funding_interval_hours,last_funding_rate\n"
        rows += "1704067200000,8,0.0001\n1704096000000,4,0.0001\n"
        client = _Client({self.URL: _FakeResponse(content=_zip(rows))})
        frame = archive.monthly_funding(client, "X", 2024, 1)
        assert list(frame["rate"]) == [0.0001, 0.0002]  # the 4h rate doubles to an 8h basis

    def test_a_missing_month_is_empty(self):
        assert archive.monthly_funding(_Client({}), "X", 2024, 1).empty


class TestSymbolEnumeration:
    def test_it_pages_the_listing_and_filters_by_quote(self):
        body = (
            "<Prefix>data/futures/um/monthly/klines/BTCUSDT/</Prefix>"
            "<Prefix>data/futures/um/monthly/klines/ETHUSDC/</Prefix>"
            "<Prefix>data/futures/um/monthly/klines/SOLUSDT/</Prefix>"
        )
        client = _Client({"listing": _FakeResponse(text=body)})
        assert archive.archived_symbols(client) == ["BTCUSDT", "SOLUSDT"]

    def test_an_empty_listing_ends_the_loop(self):
        assert archive.archived_symbols(_Client({})) == []


class TestTimezones:
    def test_naive_inputs_are_read_as_utc(self):
        assert archive._utc_ts(dt.datetime(2024, 1, 1)).tzinfo is not None

    def test_aware_inputs_are_converted(self):
        paris = pd.Timestamp("2024-01-01T01:00:00+01:00")
        assert archive._utc_ts(paris) == pd.Timestamp("2024-01-01T00:00:00Z")
