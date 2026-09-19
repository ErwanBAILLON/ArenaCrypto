# Arena

Signal-only crypto research arena. Several competing models run forward on the
same public market feed (Binance and Hyperliquid public endpoints, RSS news,
macro calendar), are scored on virtual books next to null models, improved in
the background by a champion/challenger loop, and report their convictions to
Telegram. Nothing places orders. No private API key is used for data.

Design: `docs/specs/2026-09-19-arena-design.md`. Plan: `docs/plans/`.

```bash
uv sync --extra dev
uv run pytest
uv run arena --help
```
