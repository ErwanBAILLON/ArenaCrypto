import httpx

from arena.settings import Settings
from arena.telegram.sender import send


def _settings(**kw) -> Settings:
    base = dict(database_url="", telegram_bot_token="123:abc", telegram_chat_id="42",
                dry_run=False, universe_path=None)
    base.update(kw)
    return Settings(**base)


def test_dry_run_prints_and_returns_true(capsys):
    assert send(_settings(dry_run=True), "hello") is True
    assert "[telegram dry] hello" in capsys.readouterr().out


def test_missing_token_is_dry(capsys):
    assert send(_settings(telegram_bot_token=""), "hi") is True
    assert "[telegram dry] hi" in capsys.readouterr().out


def test_http_200_returns_true():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["json"] = req.read()
        return httpx.Response(200, json={"ok": True, "result": {}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert send(_settings(), "msg", client=client) is True
    assert seen["url"] == "https://api.telegram.org/bot123:abc/sendMessage"
    assert b'"chat_id": "42"' in seen["json"] or b'"chat_id":"42"' in seen["json"]
    assert b"disable_web_page_preview" in seen["json"]


def test_http_500_returns_false_no_raise():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")))
    assert send(_settings(), "msg", client=client) is False


def test_transport_error_returns_false():
    def handler(_):
        raise httpx.ConnectError("down")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert send(_settings(), "msg", client=client) is False


def test_ok_false_returns_false():
    client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"ok": False, "description": "bad"})))
    assert send(_settings(), "msg", client=client) is False
